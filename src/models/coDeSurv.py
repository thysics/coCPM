import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import os

from torch.utils.data import DataLoader

from utils.utils import F_theta
from typing import List, Tuple

import time


class ODENet(nn.Module):
    """
    Conditional ODE Network for use in survival analysis.

    Attributes:
        output_dim (int): The output dimension of the network.
        device (torch.device): The device to use for computations.
        dudt (nn.Sequential): The sequential neural network.
        n (int): Number of points for Gauss-Legendre quadrature.
        u_n (nn.Parameter): Gauss-Legendre quadrature nodes.
        w_n (nn.Parameter): Gauss-Legendre quadrature weights.
    """

    def __init__(
        self,
        cov_dim: int,
        hidden_dim: int,
        output_dim: int,
        nonlinearity: nn.Module = nn.ReLU,
        device: str = "gpu",
        n: int = 15,
    ) -> None:
        """
        Initialize the ODENet.

        Args:
            cov_dim (int): The dimension of the covariates.
            hidden_dim (int): The dimension of the hidden layers.
            output_dim (int): The output dimension of the network.
            nonlinearity (nn.Module): The nonlinearity to use in the network.
            device (str, optional): The device to use for computations (default: 'cpu').
            n (int, optional): Number of points for Gauss-Legendre quadrature (default: 15).
        """
        super().__init__()

        self.output_dim = output_dim

        if device == "mps":
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
            print(f"FCNet: {device} specified, {self.device} used")
        elif device == "gpu":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"FCNet: {device} specified, {self.device} used")
        else:
            self.device = torch.device("cpu")
            print(f"CondODENet: {device} specified, {self.device} used")

        self.dudt = nn.Sequential(
            nn.Linear(cov_dim + 1, hidden_dim),
            nonlinearity(),
            nn.Linear(hidden_dim, hidden_dim),
            nonlinearity(),
            nn.Linear(hidden_dim, self.output_dim),
            nn.Softplus(),
        )

        for i in range(len(self.dudt)):
            nn.init.kaiming_normal_(
                self.dudt[i].weight, mode="fan_out", nonlinearity="relu"
            )

        self.n = n
        u_n, w_n = np.polynomial.legendre.leggauss(n)
        self.u_n = nn.Parameter(
            torch.tensor(u_n, device=self.device, dtype=torch.float32)[None, :],
            requires_grad=False,
        )
        self.w_n = nn.Parameter(
            torch.tensor(w_n, device=self.device, dtype=torch.float32)[None, :],
            requires_grad=False,
        )

    def mapping(self, x_: torch.Tensor) -> torch.Tensor:
        """
        Perform the mapping of input x_ through the network.

        Args:
            x_ (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The mapped output tensor.
        """
        t = x_[:, 0][:, None].to(self.device)
        x = x_[:, 1:].to(self.device)
        tau = torch.matmul(t / 2, 1 + self.u_n)  # N x n
        tau_ = torch.flatten(tau)[
            :, None
        ]  # Nn x 1. Think of as N n-dim vectors stacked on top of each other
        reppedx = torch.repeat_interleave(
            x,
            torch.tensor([self.n] * t.shape[0], dtype=torch.long, device=self.device),
            dim=0,
        )
        taux = torch.cat((tau_, reppedx), 1)  # Nn x (d+1)
        f_n = self.dudt(taux).reshape((*tau.shape, self.output_dim))  # N x n x d_out
        pred = t / 2 * ((self.w_n[:, :, None] * f_n).sum(dim=1))
        return pred

    def forward(self, x_: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x_ (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return torch.tanh(self.mapping(x_))


class ConsistentDeSurv(nn.Module):
    """
    Consistent DeSurv model for survival analysis.

    Attributes:
        device (str): The device to use for computations.
        net (CondODENet): The conditional ODE network.
        baseline: Baseline survival model.
        df_columns (List[str]): DataFrame columns.
        lr (float): Learning rate.
        optimizer (torch.optim.Optimizer): The optimizer.
    """

    def __init__(
        self,
        lr: float,
        cov_dim: int,
        hidden_dim: int,
        baseline,
        df_columns: List[str] = ["x1", "x2"],
        nonlinearity: nn.Module = nn.ReLU,
        device: str = "gpu",
        n: int = 15,
    ) -> None:
        """
        Initialize the ConsistentDeSurv model.

        Args:
            lr (float): Learning rate.
            cov_dim (int): The dimension of the covariates.
            hidden_dim (int): The dimension of the hidden layers.
            baseline: Baseline survival model. It must have a method predict_survival_function.
            df_columns (List[str], optional): DataFrame columns (default: ["x1", "x2"]).
            nonlinearity (nn.Module, optional): The nonlinearity to use in the network (default: nn.ReLU).
            device (str, optional): The device to use for computations (default: 'cpu').
            n (int, optional): Number of points for Gauss-Legendre quadrature (default: 15).
        """
        super().__init__()

        self.device = device
        self.net = ODENet(cov_dim, hidden_dim, 1, nonlinearity, self.device, n)
        self.net = self.net.to(self.device)
        self.baseline = baseline
        self.df_columns = df_columns

        self.lr = lr
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        self.loss_trace = {
            "train": {"loss": [], "regularisation": [], "likelihood": []},
            "validation": {"loss": [], "regularisation": [], "likelihood": []},
        }

    def predict(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict cumulative incidence probability.

        Args:
            x (torch.Tensor): Covariates tensor.
            t (torch.Tensor): Time tensor.

        Returns:
            torch.Tensor: Predicted cumulative incidence probability.
        """
        t = t[:, None]
        z = torch.cat((t, x), 1)
        return self.net.forward(z).squeeze()

    def predict_surv_df(self, x_test: np.ndarray, t_eval: np.ndarray) -> pd.DataFrame:
        """
        Predict survival probabilities and return a DataFrame.

        Args:
            x_test (np.ndarray): Test covariates.
            t_eval (np.ndarray): Evaluation times.

        Returns:
            pd.DataFrame: DataFrame of survival probabilities.
        """
        with torch.no_grad():
            t_ = torch.tensor(
                np.concatenate([t_eval] * x_test.shape[0], 0), dtype=torch.float32
            )
            x_ = torch.tensor(
                np.repeat(x_test, [t_eval.size] * x_test.shape[0], axis=0),
                dtype=torch.float32,
            )
            surv = pd.DataFrame(
                np.transpose(
                    (1 - self.predict(x_, t_).reshape((x_test.shape[0], t_eval.size)))
                    .detach()
                    .numpy()
                ),
                index=t_eval,
            )
            return surv

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, k: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Covariates tensor.
            t (torch.Tensor): Time tensor.
            k (torch.Tensor): Event indicator tensor.

        Returns:
            torch.Tensor: Loss value.
        """
        t = t[:, None]
        cens_ids = torch.nonzero(torch.eq(k, 0))[:, 0]
        ncens = cens_ids.size()[0]
        uncens_ids = torch.nonzero(torch.eq(k, 1))[:, 0]

        z = torch.cat((t, x), 1)
        eps = 1e-8

        censterm = 0
        if torch.numel(cens_ids) != 0:
            cdf_cens = self.net.forward(z[cens_ids, :]).squeeze()
            s_cens = 1 - cdf_cens
            censterm = torch.log(s_cens + eps).sum()

        uncensterm = 0
        if torch.numel(uncens_ids) != 0:
            cdf_uncens = self.net.forward(z[uncens_ids, :]).squeeze()
            dudt_uncens = self.net.dudt(z[uncens_ids, :]).squeeze()
            uncensterm = (
                torch.log(1 - cdf_uncens**2 + eps) + torch.log(dudt_uncens + eps)
            ).sum()

        return -(censterm + uncensterm)

    def regularisation(
        self, x: torch.Tensor, verbose: bool, n_sample: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the regularisation term
            Args:
        x (torch.Tensor): Covariates tensor.
        verbose (bool): Verbosity flag.
        n_sample (int, optional): Number of samples (default: 100).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Regularisation terms.
        """
        div = torch.zeros(size=(x.shape[0], 1))
        reg_loss = torch.zeros(size=(x.shape[0], 1))

        sample_times = torch.zeros(size=(x.shape[0] * n_sample,), device=self.device)
        sample_logp = torch.zeros(
            size=(x.shape[0] * n_sample,),
            device=self.device,
        )

        # Proposed vectorised method
        if True:
            assert n_sample > 0

            f_theta = F_theta(
                self.net, self.baseline, x, verbose, self.df_columns, self.device
            )  # Now that code is updated to take x as an argument, this doesnt need to be initialiesd every single call to regularisation()
            t_j_all = f_theta.sample_vectorised(
                x, sample_shape=n_sample
            )  # [torch.Size([50]), torch.Size([50]), torch.Size([50]), torch.Size([50]), torch.Size([50]), torch.Size([50]), torch.Size([50]), torch.Size([50]), torch.Size([50]), torch.Size([50]), torch.Size([50]), torch.Size([50]), torch.Size([50]), torch.Size([50]), torch.Size([50]), torch.Size([50]), torch.Size([50]), torch.Size([50]), torch.Size([50]), torch.Size([50]), torch.Size([50]), torch.Size([50]), torch.Size([50]), torch.Size([50]), torch.Size([50]), torch.Size([50]), torch.Size([50]), torch.Size([50]), torch.Size([50]), torch.Size([50]), torch.Size([50]), torch.Size([50]), torch.Size([50]), torch.Size([50]), torch.Size([50])]

            for i in range(x.shape[0]):
                f_theta = F_theta(
                    self.net,
                    self.baseline,
                    x[i].reshape(1, -1),
                    verbose,
                    self.df_columns,
                    self.device,
                )

                t_j = t_j_all[i]
                log_t_j = f_theta.log_prob(t_j).to(self.device)
                sample_times[i * n_sample : (i + 1) * n_sample] = t_j
                sample_logp[i * n_sample : (i + 1) * n_sample] = log_t_j

        else:
            for i in range(x.shape[0]):
                f_theta = F_theta(
                    self.net,
                    self.baseline,
                    x[i].reshape(1, -1),
                    verbose,
                    self.df_columns,
                    self.device,
                )

                # Previous method
                if True:
                    # start_time = time.time()
                    t_j = f_theta.sample(n_sample).to(self.device)  # torch.Size([50])
                    # print(f"self.regularisation sample {i} time: {time.time() - start_time:.4f} seconds")
                    # print(f"sampled time to event {torch.mean(t_j)}  {torch.std(t_j)}")
                else:
                    # New discrete empirical method
                    # start_time = time.time()
                    t_j = f_theta.sample_new(n_sample)
                    # print(f"self.regularisation sample_new {i} time: {time.time() - start_time:.4f} seconds")
                    # print(f"new sampled time to event {torch.mean(t_j)}  {torch.std(t_j)}")

                log_t_j = f_theta.log_prob(t_j).to(self.device)
                sample_times[i * n_sample : (i + 1) * n_sample] = t_j
                sample_logp[i * n_sample : (i + 1) * n_sample] = log_t_j

        # Prepare dataframe for the baseline model.
        Xi = x.repeat_interleave(n_sample, dim=0)
        df_x = pd.DataFrame(Xi.cpu().numpy(), columns=self.df_columns)

        # This code is for debugging. My eye-test suggests that obtained values are reasonable.
        #         for time, logp in zip(sample_times, sample_logp):
        #             print(f"t: {time}, p_t: {torch.exp(logp)}")

        # baseline: aft model
        F_hat_t_j = (
            1
            - torch.tensor(
                self.baseline.predict_survival_function(
                    df=df_x, times=sample_times.numpy().flatten()
                ).values,
                dtype=torch.float32,
            ).detach()
        )
        F_hat_t_j = torch.diagonal(F_hat_t_j)

        # Prevent gradients from flowing back through this part of the computational graph.
        F_t_j = self.predict(Xi, sample_times.flatten()).detach()

        F_l2 = (F_t_j - F_hat_t_j) ** 2
        xi_div = sample_logp * F_l2 + 2 * (F_t_j - F_hat_t_j) * torch.exp(sample_logp)

        div = self._average_repeated_rows(xi_div, n_sample)
        reg_loss = self._average_repeated_rows(F_l2, n_sample)

        return div.sum(), reg_loss.sum()

    def _average_repeated_rows(
        self, tensor: torch.Tensor, n_sample: int
    ) -> torch.Tensor:
        """
        Average repeated rows in a tensor.

        Args:
            tensor (torch.Tensor): Input tensor.
            n_sample (int): Number of samples.

        Returns:
            torch.Tensor: Averaged tensor.
        """
        reshaped_tensor = tensor.view(-1, n_sample, 1)
        averaged_tensor = reshaped_tensor.mean(dim=1)
        return averaged_tensor

    def compute_lik(self, x: np.ndarray, t: np.ndarray, k: np.ndarray) -> float:
        """
        Compute log likelihood for DeSurv.

        Args:
            x (np.ndarray): Covariates array.
            t (np.ndarray): Time array.
            k (np.ndarray): Event indicator array.

        Returns:
            float: Log likelihood value.
        """
        lik_mean = (
            -self.forward(
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(t, dtype=torch.float32),
                torch.tensor(k, dtype=torch.float32),
            )
            / x.shape[0]
        )

        return lik_mean.detach().item()

    def optimize(
        self,
        data_loader: DataLoader,
        n_epochs: int,
        n_sample: int = 100,
        logging_freq: int = 10,
        data_loader_val: DataLoader = None,
        max_wait: int = 20,
        pretrain_epochs: int = 0,
        lambda_: float = 1.0,
        model_state_dir=None,
        verbose: bool = True,
    ) -> None:
        """
        Optimize the model.

        Args:
            data_loader (DataLoader): Training data loader.
            n_epochs (int): Number of epochs.
            n_sample (int, optional): Number of samples (default: 100).
            logging_freq (int, optional): Logging frequency (default: 10).
            data_loader_val (DataLoader, optional): Validation data loader.
            max_wait (int, optional): Maximum wait time for early stopping (default: 20).
            pretrain_epochs (int, optional): n_epochs to be trained before adding regularisation term.
            lambda_ (float, optional): hyparameter to control the impact of reguluarisation term.
            verbose (bool, optional): Verbosity flag (default: True).
        """
        if data_loader_val is not None:
            best_val_reg_loss = np.inf
            wait = 0

        if model_state_dir is None:
            model_state_dir = (
                os.path.dirname(os.path.realpath(__file__)) + "/../../eval/"
            )
            print(model_state_dir)

        for epoch in range(n_epochs):
            train_loss = 0.0
            lik_loss = 0.0
            reg_loss = 0.0

            for batch_idx, (x, t, k, o) in enumerate(data_loader):
                argsort_t = torch.argsort(t)
                x_ = x[argsort_t, :].to(self.device)
                t_ = t[argsort_t].to(self.device)
                k_ = k[argsort_t].to(self.device)
                o_ = o[argsort_t].to(self.device)

                # Split x_ into in-distribution and out-of-distribution
                x_in = x_[o_ == 0.0]
                t_in = t_[o_ == 0.0]
                k_in = k_[o_ == 0.0]

                x_ood = x_[o_ == 1.0]
                t_ood = t_[o_ == 1.0]
                k_ood = k_[o_ == 1.0]

                # start_time = time.time()  # Start timer
                self.optimizer.zero_grad()
                likelihood_term = self.forward(x_in, t_in, k_in)
                # print(f"self.forward time: {time.time() - start_time:.4f} seconds")

                consistency_term = 0.0
                reg_loss_term = 0.0

                if epoch >= pretrain_epochs:
                    if x_ood.shape[0] > 0:
                        # start_time = time.time()  # Start timer
                        consistency_term, reg_loss_term_ = self.regularisation(
                            x=x_ood,
                            n_sample=n_sample,
                            verbose=verbose,
                        )
                        reg_loss_term = reg_loss_term_
                        # print(f"self.regularisation time: {time.time() - start_time:.4f} seconds")

                loss = likelihood_term + lambda_ * consistency_term

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                lik_loss += likelihood_term.item()
                reg_loss += reg_loss_term

            self.loss_trace["train"]["loss"].append(train_loss)
            self.loss_trace["train"]["regularisation"].append(reg_loss)
            self.loss_trace["train"]["likelihood"].append(lik_loss)

            if epoch % logging_freq == 0:
                # if verbose:
                #    print(f"\tEpoch: {epoch:2}. Total loss: {train_loss:11.2f}")
                #    print(f"\tEpoch: {epoch:2}. Regularisation: {reg_loss:11.2f}")

                if epoch == pretrain_epochs:
                    print(f"Adding regularisation term with lambda {lambda_}")

                if data_loader_val is not None:
                    val_loss = 0.0
                    val_lik_loss = 0.0
                    val_reg_loss = 0.0
                    val_cons_loss = 0.0
                    for batch_idx, (x, t, k, o) in enumerate(data_loader_val):
                        argsort_t = torch.argsort(t)
                        x_ = x[argsort_t, :].to(self.device)
                        t_ = t[argsort_t].to(self.device)
                        k_ = k[argsort_t].to(self.device)
                        o_ = o[argsort_t].to(self.device)

                        # Split x_ into in-distribution and out-of-distribution
                        x_in = x_[o_ == 0.0]
                        t_in = t_[o_ == 0.0]
                        k_in = k_[o_ == 0.0]

                        x_ood = x_[o_ == 1.0]
                        t_ood = t_[o_ == 1.0]
                        k_ood = k_[o_ == 1.0]

                        val_likelihood_term = self.forward(x_in, t_in, k_in)

                        val_consistency_term = 0.0
                        val_reg_loss_term = 0.0

                        if epoch >= pretrain_epochs:
                            if x_ood.shape[0] > 0:
                                val_consistency_term, val_reg_loss_term = (
                                    self.regularisation(
                                        x=x_ood, n_sample=n_sample, verbose=verbose
                                    )
                                )

                        loss = val_likelihood_term + lambda_ * val_consistency_term

                        val_loss += loss.item()
                        val_lik_loss += val_likelihood_term.item()
                        val_reg_loss += val_reg_loss_term
                        val_cons_loss += val_consistency_term

                    self.loss_trace["validation"]["loss"].append(val_loss)
                    self.loss_trace["validation"]["regularisation"].append(val_reg_loss)
                    self.loss_trace["validation"]["likelihood"].append(val_lik_loss)

                    if epoch >= pretrain_epochs:
                        if val_reg_loss < best_val_reg_loss:
                            best_val_reg_loss = val_reg_loss
                            wait = 0
                            torch.save(
                                self.state_dict(), model_state_dir + "codesurv_low"
                            )
                        else:
                            wait += 1

                        if wait > max_wait:
                            state_dict = torch.load(model_state_dir + "codesurv_low")
                            self.load_state_dict(state_dict)
                            return

                    if verbose:
                        print(
                            f"\tEpoch: {epoch:2}{'*' if wait == 0 else ''}. Total train loss: {train_loss:11.2f}. Total val loss: {val_loss:11.2f}. Likelihood+Penalty: ({val_lik_loss:11.2f}+{lambda_}*{val_cons_loss:11.2f}) ... (with regularisation: {val_reg_loss:11.2f})"
                        )

        if data_loader_val is not None:
            state_dict = torch.load(model_state_dir + "codesurv_low")
            self.load_state_dict(state_dict)
