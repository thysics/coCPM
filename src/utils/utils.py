import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import softplus

from torch.distributions import Distribution
from scipy.optimize import root_scalar

from typing import Union, List, Tuple

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
    
    def __init__(self, net: nn.Module, baseline, x: torch.Tensor, verbose: bool, df_columns: list[str], device: str = "cpu") -> None:
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

    def _compute_cdf(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the cumulative distribution function (CDF) for a given t.

        Args:
            t (torch.Tensor): Input tensor of shape (N,).

        Returns:
            torch.Tensor: The computed CDF.
        """
        t = t[:, None].to(self.device)
        z = torch.cat((t, self.x), 1)
        res = self.net.forward(z).squeeze()
        return res

    def _inverse_cdf(self, y: torch.Tensor, t_max: float = 500.) -> float:
        """
        Compute the inverse CDF for a given y using root_scalar.

        Args:
            y (torch.Tensor)

        Returns:
            float: The inverse CDF value.
        """
#         t_sample = self._compute_cdf(y)
        
     
        result = root_scalar(
            lambda t: self._compute_cdf(torch.tensor([t], device=self.device, dtype=torch.float32)).detach().numpy() - y.detach().numpy(),
            bracket=[self.t_min, t_max],
            method='bisect'
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
        
        m = torch.distributions.uniform.Uniform(torch.tensor([0.0001]), torch.tensor([0.950]))
        value = m.sample().detach()
        t_max = 1.0
        
        while retries < max_retries:    
            try:
                return self._inverse_cdf(value, t_max)
            except Exception as e:
                F_t_max = self._compute_cdf(torch.tensor([t_max], device = self.device, dtype = torch.float32))
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
                    raise RuntimeError(f"Max retries reached for sample {value} with t_max: {t_max}")

    def sample(self, sample_shape: Union[torch.Size, int] = torch.Size()) -> torch.Tensor:
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