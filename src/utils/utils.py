import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import root_scalar
from typing import Union


class F_theta(nn.Module):
    """
    Distribution class that represents F_theta.

    Attributes:
        net (nn.Module): The neural network model.
        x (torch.Tensor): Input tensor.
        t_min (float): Minimum value of t.
        verbose (bool): Verbosity flag.
        df_columns (list[str]): list[column_names] that is used to train the baseline model.
        device (str): The device to use for computations (default: 'cpu').
    """

    def __init__(
        self,
        net: nn.Module,
        baseline,
        x: torch.Tensor,
        verbose: bool,
        df_columns: list[str],
        device: str = "cpu",
    ) -> None:
        """
        Initialize the F_theta class.

        Args:
            net (nn.Module): The neural network model.
            baseline: The baseline model (e.g., AFT)
            x (torch.Tensor): Input tensor.
            verbose (bool): Verbosity flag.
            device (str, optional): The device to use for computations (default: 'cpu').
        """
        super(F_theta, self).__init__()
        self.device = device
        self.baseline = baseline
        self.net = net.to(device)
        self.x = x
        self.t_min = 0
        self.verbose = verbose
        self.df_columns = df_columns

    def _compute_cdf(self, t: torch.Tensor, x=None) -> torch.Tensor:
        """
        Compute the cumulative distribution function (CDF) for a given t.

        Args:
            t (torch.Tensor): Input tensor of shape (N,).

        Returns:
            torch.Tensor: The computed CDF.
        """
        if x is None:
            x = self.x

        # print(x.shape)   # torch.Size([3200, 2])   torch.Size([1, 2])
        # print(t.shape)   # torch.Size([3200])      torch.Size([1])
        t = t[:, None].to(self.device)
        z = torch.cat((t, x), 1)
        return self.net.forward(z).squeeze()

    def _inverse_transform_sample(self, t_min: float = 0, t_max: float = 1.0):
        """
        Use Inverse Transform Sampling on a grid

        Note: In simulated example the data isnt standardised to [0,1] across all datasets - as the maximum duration in D2 and D3 is greater than in D1
        """
        t_eval = np.linspace(t_min, t_max, 100)

        with torch.no_grad():
            # TODO: vectorise this
            pred = []
            for _t in t_eval:
                _t = torch.tensor([_t], device=self.device, dtype=torch.float32)
                _pred = self._compute_cdf(_t).item()
                pred.append(_pred)

        rsample = np.random.uniform(
            0, pred[-1]
        )  # Randomly sample between 0 and the maximum cumulative prob F(t_max)
        time_index = np.sum(np.asarray(pred) <= rsample) - 1

        return torch.tensor(
            [t_eval[time_index]], device=self.device, dtype=torch.float32
        )

    def sample_new(
        self, sample_shape: Union[torch.Size, int] = torch.Size()
    ) -> torch.Tensor:
        """ """

        samples = [self._inverse_transform_sample() for _ in range(sample_shape)]
        return torch.tensor(samples, device=self.device)

    def _inverse_transform_sample_vectorised(
        self, x, t_min: float = 0, t_max: float = 1.0
    ) -> torch.Tensor:
        """ """

        self.t_eval = np.linspace(t_min, t_max, 100)

        t_test = torch.tensor(
            np.concatenate([self.t_eval] * x.shape[0], 0),
            dtype=torch.float32,
            device=self.device,
        )
        X_test = x.repeat_interleave(self.t_eval.size, 0).to(self.device, torch.float32)

        # Batched predict: Cannot make all predictions at once due to memory constraints
        pred_bsz = 2**15  # Predict in batches
        pred = []
        pi = []
        for X_test_batched, t_test_batched in zip(
            torch.split(X_test, pred_bsz), torch.split(t_test, pred_bsz)
        ):
            _pred = self._compute_cdf(t_test_batched, x=X_test_batched)
            pred.append(_pred)
        pred = torch.concat(pred).reshape(x.shape[0], self.t_eval.size)
        # print(pred)   # torch.Size([35, 100])

        # Normalize to cumulative probability (assuming increasing order along seq_len)
        min_vals = pred[:, 0].unsqueeze(1)  # (bsz, 1)
        max_vals = pred[:, -1].unsqueeze(1)  # (bsz, 1)

        # Sample a uniform random number in the range of each sequence
        sampled_values = (
            torch.rand(pred.shape[0], 1) * (max_vals - min_vals) + min_vals
        )  # (bsz, 1)

        # Find the index of the closest value in the sequence
        indices = torch.abs(pred - sampled_values).argmin(dim=1)  # (bsz,)

        time_points = torch.tensor(
            self.t_eval[indices], device=self.device, dtype=torch.float32
        )  # (bsz,)

        return time_points

    def sample_vectorised(
        self, x, sample_shape: int = 50, t_min: float = 0.0, t_max: float = 1.0
    ):
        """ """
        sample_list = [
            self._inverse_transform_sample_vectorised(x, t_min=t_min, t_max=t_max)
            for _ in range(sample_shape)
        ]  # list where each tensor is of shape (bsz,)

        sample_list = torch.stack(sample_list, dim=0)  # Shape: (sample_shape, bsz,)

        # catch cases where x.shape==torch.Size([1, 2])
        if len(sample_list.shape) != 2:
            sample_list = sample_list.unsqueeze(-1)
            # print(x.shape)
            # print(x)
            # print(sample_list)
            # print(sample_shape)
            # print(sample_list.shape)

        sample_list = list(
            sample_list.permute(1, 0)
        )  # Convert back to a list of length bsz, with each element being a vector of length sample_shape

        return sample_list

    def _inverse_cdf(self, y: torch.Tensor, t_max: float = 500.0) -> float:
        """
        Compute the inverse CDF for a given y using root_scalar.

        Args:
            y (torch.Tensor)

        Returns:
            float: The inverse CDF value.
        """
        #         t_sample = self._compute_cdf(y)

        result = root_scalar(
            lambda t: self._compute_cdf(
                torch.tensor([t], device=self.device, dtype=torch.float32)
            )
            .detach()
            .numpy()
            - y.detach().numpy(),
            bracket=[self.t_min, t_max],
            method="bisect",
        )

        return result.root

    def _inverse_cdf_with_retries(self, max_retries: int = 30) -> float:
        """
        Compute the inverse CDF with retries in case of errors.

        Args:
            max_retries (int, optional): Maximum number of retries (default: 10).

        Returns:
            float: The inverse CDF value.
        """
        retries = 0

        m = torch.distributions.uniform.Uniform(
            torch.tensor([0.0001]), torch.tensor([0.950])
        )
        value = m.sample().detach()
        t_max = 1.0

        while retries < max_retries:
            try:
                return self._inverse_cdf(value, t_max)
            except Exception as e:
                F_t_max = self._compute_cdf(
                    torch.tensor([t_max], device=self.device, dtype=torch.float32)
                )
                #                 print(f"Error occurred with uniform r.v: {value} and t_max: {t_max}, F_t_max: {F_t_max}. Retrying...")
                t_max += 20
                #                 value = m.sample().detach()
                retries += 1

                if retries == 15:
                    value = m.sample().detach()
                    t_max = 3.0

                if retries == max_retries:
                    print(f"F_t_max: {F_t_max}, current_value: {value}")
                    #                     print(f"Error occurred with uniform r.v: {value} and t_max: {t_max}. Retrying...")
                    raise RuntimeError(
                        f"Max retries reached for sample {value} with t_max: {t_max}"
                    )

    def sample(
        self, sample_shape: Union[torch.Size, int] = torch.Size()
    ) -> torch.Tensor:
        """
        Generate samples from the distribution.

        Args:
            sample_shape (Union[torch.Size, int], optional): Shape of the samples (default: torch.Size()).

        Returns:
            torch.Tensor: The generated samples.
        """
        samples = [self._inverse_cdf_with_retries() for _ in range(sample_shape)]
        return torch.tensor(samples, device=self.device)

    def log_prob(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the log probability of t.

        Args:
            t (torch.Tensor): Input tensor of shape (N,).

        Returns:
            torch.Tensor: The log probability.
        """
        t = t[:, None].to(self.device)
        if t.shape[0] > 1:
            x = self.x.repeat(t.shape[0], 1)
        else:
            x = self.x

        z = torch.cat((t, x), 1)
        eps = 1e-8

        cdf_uncens = self.net.forward(z).squeeze()
        dudt_uncens = self.net.dudt(z).squeeze()
        log_p_tx = torch.log(1 - cdf_uncens**2 + eps) + torch.log(dudt_uncens + eps)

        return log_p_tx
