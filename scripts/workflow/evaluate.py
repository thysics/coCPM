import torch
import torch.nn as nn
import os

import pandas as pd
from collections import defaultdict

from sksurv.metrics import concordance_index_censored

from torch.utils.data import TensorDataset, DataLoader
from lifelines import WeibullAFTFitter

from models.coDeSurv import ConsistentDeSurv
from models.DeSurv import DeSurv
from utils.helpers import split_dataframe, split_df, predict_cif, compute_timediff


class Eval:
    def __init__(
        self,
        train_data_path: str,
        train_data_star_path: str,
        test_data_path: str,
        lr: float,
        hidden_dim: int,
    ):
        self.train_data = pd.read_csv(train_data_path)  # True OOD
        self.test_data = pd.read_csv(test_data_path)  # Test Set
        self.train_data_star = pd.read_csv(train_data_star_path)  # Simulated OOD

        self.seed_prefix = train_data_path[-7:-4]
        self.model_state_dir = (
            os.path.dirname(os.path.realpath(__file__))
            + "/../../eval/"
            + self.seed_prefix
        )

        self.x_label = ["x1", "x2"]
        self.t_label = "Duration"
        self.e_label = "Censor"
        self.o_label = "OOD"

        print("Training baseline AFT model using D1")
        self.aft = self._train_aft(self.train_data[self.train_data.OOD == 0.0])
        self.desurv = self._set_desurv(self.aft, lr=lr, hidden_dim=hidden_dim)
        self.codesurv = self._set_codesurv(self.aft, lr=lr, hidden_dim=hidden_dim)

    def get_loader(
        self, df: pd.DataFrame, batch_size: int, drop_last: bool = False
    ) -> DataLoader:
        dataset_train = self._get_dataset(df)

        return DataLoader(
            dataset_train,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=drop_last,
        )

    def _get_dataset(self, df: pd.DataFrame) -> TensorDataset:
        x_train, t_train, e_train, o_train = split_df(
            df, self.x_label, self.t_label, self.e_label, self.o_label
        )
        return TensorDataset(
            *[
                torch.tensor(u, dtype=dtype_)
                for u, dtype_ in [
                    (x_train, torch.float32),
                    (t_train, torch.float32),
                    (e_train, torch.long),
                    (o_train, torch.long),
                ]
            ]
        )

    def _train_aft(self, train_data: pd.DataFrame) -> WeibullAFTFitter:
        # Train baseline (AFT)
        aft = WeibullAFTFitter()
        aft.fit(
            train_data[[self.t_label] + [self.e_label] + self.x_label],
            duration_col=self.t_label,
            event_col=self.e_label,
            show_progress=True,
        )
        return aft

    def _set_desurv(self, baseline, lr: float, hidden_dim: int) -> DeSurv:
        return DeSurv(
            lr,
            len(self.x_label),
            hidden_dim,
            baseline=baseline,
            nonlinearity=nn.ReLU,
            device="cpu",
            n=15,
            df_columns=self.x_label,
        )

    def _train_desurv(
        self,
        train_data,
        batch_size: int,
        n_epochs: int,
        max_wait: int,
    ) -> None:
        # Train DeSurv
        df_train_d1, df_val_d1 = split_dataframe(train_data)

        d1_loader_train = self.get_loader(df_train_d1, batch_size)
        d1_loader_val = self.get_loader(df_val_d1, batch_size)

        self.desurv.optimize(
            d1_loader_train,
            n_epochs=n_epochs,
            logging_freq=5,  # 10,
            data_loader_val=d1_loader_val,
            max_wait=max_wait,
            model_state_dir=self.model_state_dir,
        )

        torch.save(self.desurv.state_dict(), self.model_state_dir + "eval_desurv")

    def _set_codesurv(self, baseline, lr: float, hidden_dim: int) -> ConsistentDeSurv:
        return ConsistentDeSurv(
            lr,
            len(self.x_label),
            hidden_dim,
            baseline=baseline,
            nonlinearity=nn.ReLU,
            device="cpu",
            n=15,
            df_columns=self.x_label,
        )

    def _train_codesurv(
        self,
        train_data: pd.DataFrame,
        batch_size: int,
        oracle: bool,
        lamda: float,
        n_epochs: int,
        max_wait: int,
    ) -> None:
        # Train coDeSurv
        df_train, df_val = split_dataframe(train_data)

        data_loader_train = self.get_loader(
            df=df_train, batch_size=batch_size, drop_last=True
        )
        data_loader_val = self.get_loader(df_val, batch_size)

        self.codesurv.optimize(
            data_loader_train,
            n_sample=100,
            n_epochs=n_epochs,  #
            logging_freq=5,  # 10,
            data_loader_val=data_loader_val,
            max_wait=max_wait,
            lambda_=lamda if not oracle else 1.0,
            pretrain_epochs=100,
            model_state_dir=self.model_state_dir,
            verbose=True,
        )

        if oracle:
            torch.save(
                self.codesurv.state_dict(),
                self.model_state_dir + f"eval_codesurv_oracle",
            )
        else:
            torch.save(
                self.codesurv.state_dict(),
                self.model_state_dir + f"eval_codesurv_{lamda}",
            )

    def train(
        self,
        batch_size: int,
        n_epochs: int = 200,
        max_wait: int = 100,
        lambdas: list[float] = [0.1, 1.0],
    ) -> None:
        # Train baseline & DeSurv using D1 data
        d1_data = self.train_data[self.train_data.OOD == 0.0]

        print("Training DeSurv model using D1")
        self._train_desurv(d1_data, batch_size, n_epochs, max_wait)

        # Train coDeSurv using D1 and D2
        print("Training coDeSurv model using D1 and D2")
        self._train_codesurv(
            self.train_data,
            batch_size,
            True,
            -1.0,
            n_epochs,
            max_wait,
        )

        # Train coDeSurv using D1 and D3
        print("Training coDeSurv model using D1 and D3")
        for lamda in lambdas:
            print(f"\tlambda {lamda}")
            self._train_codesurv(
                self.train_data_star,
                batch_size,
                False,
                lamda,
                n_epochs,
                max_wait,
            )

    def evaluate(
        self, iterations: int, lambdas: list[float], batch_size: int, n_sample: int
    ) -> dict:
        data_ood_test = self.test_data[self.test_data.OOD == 1.0]
        data_id_test = self.test_data[self.test_data.OOD == 0.0]

        data_loader_test = self.get_loader(self.test_data, batch_size)

        benchmark = {
            f"coDeSurv_lambda_{str(lam_).replace('.', '_')}": defaultdict(list)
            for lam_ in lambdas
        }
        models = {
            "DeSurv": defaultdict(list),
            "coDeSurv_Oracle": defaultdict(list),
            "baseline": defaultdict(list),
            **benchmark,
        }

        # DeSurv
        for i in range(iterations):
            state_dict = torch.load(self.model_state_dir + f"eval_desurv")
            self.desurv.load_state_dict(state_dict)
            self.desurv.eval()
            inconsistency_metric = compute_timediff(
                self.desurv,
                data_loader_test,
                n_sample=n_sample,
                df_type="D2",
                verbose=False,
            )
            models["DeSurv"][f"consistency"].append(inconsistency_metric)

            # coDeSurv (D1 and D2)
            state_dict = torch.load(self.model_state_dir + f"eval_codesurv_oracle")
            self.codesurv.load_state_dict(state_dict)
            self.codesurv.eval()
            inconsistency_metric = compute_timediff(
                self.codesurv,
                data_loader_test,
                n_sample=n_sample,
                df_type="D2",
                verbose=False,
            )
            models["coDeSurv_Oracle"][f"consistency"].append(inconsistency_metric)

            # CoDeSurv (D1 and D3)
            for lamda in lambdas:
                state_dict = torch.load(self.model_state_dir + f"eval_codesurv_{lamda}")
                self.codesurv.load_state_dict(state_dict)
                self.codesurv.eval()
                inconsistency_metric = compute_timediff(
                    self.codesurv,
                    data_loader_test,
                    n_sample=n_sample,
                    df_type="D2",
                    verbose=False,
                )
                models[f"coDeSurv_lambda_{str(lamda).replace('.', '_')}"][
                    f"consistency"
                ].append(inconsistency_metric)

        def compute_cidx(model_name: str, df: pd.DataFrame):
            res = 0
            if model_name == "baseline":
                res = concordance_index_censored(
                    df["Censor"] == 1.0, df["Duration"], -self.aft.predict_median(df)
                )
            elif model_name == "DeSurv":
                res = concordance_index_censored(
                    df["Censor"] == 1.0,
                    df["Duration"],
                    -predict_cif(self.desurv, df, self.x_label, self.t_label)
                    .detach()
                    .numpy(),
                )
            elif model_name == "coDeSurv_Oracle":
                state_dict = torch.load(self.model_state_dir + f"eval_codesurv_oracle")
                self.codesurv.load_state_dict(state_dict)
                self.codesurv.eval()

                res = concordance_index_censored(
                    df["Censor"] == 1.0,
                    df["Duration"],
                    -predict_cif(self.codesurv, df, self.x_label, self.t_label)
                    .detach()
                    .numpy(),
                )
            else:  # specify the value of lambda, instead of model name
                state_dict = torch.load(
                    self.model_state_dir + f"eval_codesurv_{model_name}"
                )
                self.codesurv.load_state_dict(state_dict)
                self.codesurv.eval()

                res = concordance_index_censored(
                    df["Censor"] == 1.0,
                    df["Duration"],
                    -predict_cif(self.codesurv, df, self.x_label, self.t_label)
                    .detach()
                    .numpy(),
                )

            return res

        for model_name in ["baseline", "DeSurv", "coDeSurv_Oracle"] + lambdas:
            cidx_d1, cidx_d2 = (
                compute_cidx(model_name, data_id_test),
                compute_cidx(model_name, data_ood_test),
            )
            if model_name in ["baseline", "DeSurv", "coDeSurv_Oracle"]:
                models[model_name]["c-index_d1"].append(cidx_d1[0])
                models[model_name]["c-index_d2"].append(cidx_d2[0])
            else:
                models[f"coDeSurv_lambda_{str(model_name).replace('.', '_')}"][
                    f"c-index_d1"
                ].append(cidx_d1[0])
                models[f"coDeSurv_lambda_{str(model_name).replace('.', '_')}"][
                    f"c-index_d2"
                ].append(cidx_d2[0])

        return models
