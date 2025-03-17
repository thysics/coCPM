#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data generation script for survival analysis with OOD examples.

This script replicates the exact data generation process from the notebook.
It creates:
1. H1 data: Covariates from standard normal distribution
2. H2 data: Covariates from shifted normal distribution
3. In-distribution (ID) and out-of-distribution (OOD) splits based on PDF values
4. Survival times with censoring
5. Preprocessed datasets for training and testing

Usage:
    python data_generation.py --n 1000 --seed 13 --path './data'
"""

import numpy as np
import pandas as pd
import os
import argparse
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from typing import Tuple


class DGP:
    """
    Weibull Accelerated Failure Time (AFT) model for generating synthetic data.

    Methods:
        gumbel_samples(N: int) -> np.ndarray:
            Generate Gumbel(0,1) samples.

        generate_covariates(type: str, N: int) -> np.ndarray:
            Generate covariates from a specific distribution.

        generate_survival_time(X: np.ndarray, k: float, weights: np.ndarray,
                              censoring_rate: float, type: str) -> Tuple[np.ndarray, np.ndarray]:
            Generate survival times and censoring indicators.
    """

    def __init__(self) -> None:
        pass

    def gumbel_samples(self, N: int) -> np.ndarray:
        """
        Generate Gumbel(0,1) samples.

        Args:
            N (int): Number of samples to generate.

        Returns:
            np.ndarray: Array of Gumbel(0,1) samples.
        """
        # Generate uniform random samples
        U = np.random.uniform(size=N)

        # Transform to Gumbel(0,1) distribution
        return -np.log(-np.log(U))

    def log_normal_samples(self, N: int, mean: float, sigma: float) -> np.ndarray:
        """
        Generate log-normal samples.

        Args:
            N (int): Number of samples to generate.
            mean (float): Mean of the underlying normal distribution.
            sigma (float): Standard deviation of the underlying normal distribution.

        Returns:
            np.ndarray: Array of log-normal samples.
        """
        # Generate normal random samples
        normal_samples = np.random.normal(loc=mean, scale=sigma, size=N)

        # Transform to log-normal distribution
        log_normal_samples = np.exp(normal_samples)

        return log_normal_samples

    def generate_covariates(self, type: str = "H1", N: int = 1000):
        """
        Generate covariates based on the specified type.

        Args:
            type (str): "H1" for standard normal distribution, "H2" for shifted distribution
            N (int): Number of samples to generate

        Returns:
            np.ndarray: Generated covariates of shape (N, 2)
        """
        if type == "H1":
            res = np.random.normal(0, 1, size=(N, 2))
        else:
            # Define the mean vector
            mean_ = [-1, 1]

            # Define the covariance matrix
            cov_matrix = np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            )

            # Generate samples from the multivariate normal distribution
            res = np.random.multivariate_normal(mean_, cov_matrix, size=N)

        return res

    def generate_survival_time(
        self,
        X: np.ndarray,
        k: float,
        weights: np.ndarray,
        censoring_rate: float,
        random_seed: int = 13,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate survival times based on the given covariates.

        Args:
            X (np.ndarray): Covariate matrix of shape (N, 2)
            k (float): Scale parameter
            weights (np.ndarray): Weight vector for the model
            censoring_rate (float): Proportion of censored observations

        Returns:
            tuple: (survival_times, event_indicators)
        """

        np.random.seed(random_seed)

        N = X.shape[0]
        Z = self.gumbel_samples(N=N)
        Y = np.zeros(N)
        event_indicator = np.ones(N)

        assert len(weights) == 4, "Weights must have 4-dimension."

        for i in range(N):
            ln_Y = (
                (weights[:2] * X[i]).sum()
                + weights[2] * (X[i, 0] * X[i, 1])
                + weights[3] * (X[i, 0] + X[i, 1] - 0.5) ** 2
                + 1 / k * Z[i]
            )
            Y[i] = np.exp(ln_Y)

        if censoring_rate > 0:
            num_censored = int(N * censoring_rate)
            censored_indices = np.random.choice(N, num_censored, replace=False)

            for idx in censored_indices:
                event_indicator[idx] = 0
                Y[idx] = np.random.uniform(0, Y[idx])

        return Y, event_indicator


def compute_pdf(data):
    """
    Compute the probability density function (PDF) values for a given N x 2 numpy array.
    Each row of the array is assumed to be drawn from a bivariate normal distribution with mean 0 and variance 1.

    Parameters:
    data (numpy.ndarray): N x 2 array where each row represents a 2D point.

    Returns:
    numpy.ndarray: N-dimensional array containing the PDF values for each point.
    """
    # Define the mean and covariance matrix for the bivariate normal distribution
    mean = np.array([0, 0])  # Mean vector [0, 0]
    cov = np.array(
        [[1, 0], [0, 1]]
    )  # Covariance matrix (identity matrix for independent variables)

    # Create a multivariate normal distribution object
    rv = multivariate_normal(mean, cov)

    # Compute the PDF values for each point in the input data
    pdf_values = rv.pdf(data)

    return pdf_values


def create_ood_indicator(arr, pdf=True, threshold=None):
    """
    Takes an N x D numpy array and returns a binary N x 1 numpy array.
    Each entry is 0 if the Euclidean distance of the n-th dimension is less than the 90th percentile,
    and 1 if it exceeds that.

    Parameters:
        arr (numpy.ndarray): An N x D numpy array.

    Returns:
        numpy.ndarray: An N x 1 binary numpy array.
    """
    if pdf:
        distances = compute_pdf(arr)
    else:
        # Compute the Euclidean distance of each row from the origin
        distances = np.linalg.norm(arr, axis=1)

    if threshold is None:
        # Find the 90th percentile of the distances
        threshold = np.percentile(distances, 90)

    if pdf:
        binary_array = (distances <= threshold).astype(int)
    else:
        # Create a binary array based on the threshold
        binary_array = (distances >= threshold).astype(int)

    return binary_array


def drop_high_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove outliers in the duration column (greater than 99th percentile).

    Args:
        df (pd.DataFrame): Input dataframe with 'Duration' column

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    # Calculate the 99th percentile for the 'duration' column
    percentile_99 = df["Duration"].quantile(0.99)

    # Filter out rows where 'duration' is greater than the 99th percentile
    df_filtered = df[df["Duration"] <= percentile_99]

    return df_filtered


def normalize_column(df, ref_df, column_name):
    """
    Normalize the specified column in the DataFrame to be between 0 and 1.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    ref_df (pd.DataFrame): The reference DataFrame.
    column_name (str): The column to be normalized.

    Returns:
    pd.DataFrame: The DataFrame with the normalized column.
    """
    df = df.copy()  # To avoid modifying the original DataFrame
    min_value = ref_df[column_name].min()
    max_value = ref_df[column_name].max()
    df[column_name] = (df[column_name] - min_value) / (max_value - min_value)
    df[column_name] += 1e-8  # Add small value to avoid zeros
    return df


def _split_base_to_ood_id(x_ood_base, x_id_base):
    # Get threshold for OOD detection - 5th percentile of PDF values from H1
    print("Computing PDF threshold for OOD detection...")
    thres = np.percentile(compute_pdf(x_id_base), 5)

    # Identify OOD samples in H2
    ood_indicator = create_ood_indicator(x_ood_base, pdf=True, threshold=thres)

    # Split H2 into OOD and ID parts
    return x_ood_base[ood_indicator == 1.0], x_ood_base[ood_indicator == 0.0]


def _gen_dataset(
    dgp,
    x_cov,
    random_seed: int,
    weights: list = [0.2, 0.5, -0.1, -0.4],
    censoring_rate: float = 0.3,
    k: int = 3,
):
    time_to_event, censoring_ind = dgp.generate_survival_time(
        X=x_cov,
        k=k,
        weights=weights,
        censoring_rate=censoring_rate,
        random_seed=random_seed,
    )
    return pd.DataFrame(
        np.column_stack([time_to_event, censoring_ind, x_cov]),
        columns=["Duration", "Censor", "x1", "x2"],
    )


def _gen_ood_star(dgp: DGP, random_seed: int, n: int):
    np.random.seed(random_seed)

    x_ood_star_base = dgp.generate_covariates(type="H1", N=20 * n)

    thres = np.percentile(compute_pdf(x_ood_star_base), 1)
    ood_indicator = create_ood_indicator(x_ood_star_base, threshold=thres)
    x_ood_star = x_ood_star_base[ood_indicator == 1.0]

    train_idx = np.random.choice(np.arange(x_ood_star.shape[0]), 203)

    x_ood_star = x_ood_star[train_idx]

    return _gen_dataset(dgp=dgp, x_cov=x_ood_star, random_seed=random_seed)


def generate_data(n=1000, random_seed=13, file_path="./data"):
    """
    Generate synthetic data for survival analysis, following the exact process in the notebook.

    Args:
        n (int): Number of samples to generate for H1 distribution
        random_seed (int): Random seed for reproducibility
        file_path (str): Path to save the generated datasets

    Returns:
        None (saves files to disk)
    """
    # Set random seed
    np.random.seed(random_seed)

    # Define column labels
    t_label = "Duration"
    o_label = "OOD"

    # Initialize data generator
    dgp = DGP()

    # Generate covariates
    print(f"Generating {n} samples from H1 distribution...")
    x_id_base = dgp.generate_covariates(type="H1", N=n)

    print(f"Generating {2 * n} samples from H2 distribution...")
    x_ood_base = dgp.generate_covariates(type="H2", N=2 * n)

    x_ood, x_id_test = _split_base_to_ood_id(x_ood_base=x_ood_base, x_id_base=x_id_base)

    # Define model weights for G2 model

    # Create dataframes
    print("Creating dataframes...")

    data_train = _gen_dataset(dgp=dgp, x_cov=x_id_base, random_seed=random_seed)
    data_ood_base = _gen_dataset(dgp=dgp, x_cov=x_ood, random_seed=random_seed)
    data_id_test_base = _gen_dataset(dgp=dgp, x_cov=x_id_test, random_seed=random_seed)
    data_ood_star = _gen_ood_star(dgp=dgp, random_seed=random_seed, n=n)

    # Split OOD data into training and test sets
    data_ood_train, data_ood_test = train_test_split(
        data_ood_base, test_size=0.6, train_size=0.4, random_state=random_seed
    )

    # Sample ID test to match OOD test size
    data_id_test = data_id_test_base.sample(
        data_ood_test.shape[0], random_state=random_seed
    )

    # Add OOD indicator
    data_id_test[o_label] = 0.0
    data_ood_test[o_label] = 1.0

    # Combine test datasets
    test_data = pd.concat([data_id_test, data_ood_test], axis=0)

    # Preprocess data
    print("Preprocessing data...")
    # d1_data = data_train[data_train.OOD == 0.0]
    # d2_data = data_train[data_train.OOD == 1.0]
    d1_data = drop_high_duration(data_train)
    d2_data = drop_high_duration(data_ood_train)
    d3_data = drop_high_duration(data_ood_star)

    # Normalize durations
    d1_data = normalize_column(d1_data, d1_data, t_label)
    d2_data = normalize_column(d2_data, d1_data, t_label)
    d3_data = normalize_column(d3_data, d1_data, t_label)

    # Add OOD indicators
    d1_data[o_label] = 0.0
    d2_data[o_label] = 1.0
    d3_data[o_label] = 1.0

    # Combine training datasets
    train_data = pd.concat([d1_data, d2_data], axis=0)
    train_data_star = pd.concat([d1_data, d3_data], axis=0)

    # Create output directory if it doesn't exist
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # Save datasets
    print(f"Saving datasets to {file_path}...")
    train_data.to_csv(os.path.join(file_path, f"sim_train.csv"), index=False)
    train_data_star.to_csv(os.path.join(file_path, f"sim_train_star.csv"), index=False)
    test_data.to_csv(os.path.join(file_path, "sim_test.csv"), index=False)

    # Print dataset info
    print("\nDataset Information:")
    print(
        f"Training set: {train_data.shape[0]} samples, {train_data.shape[1]} features"
    )
    print(
        f"Training set (star): {train_data_star.shape[0]} samples, {train_data_star.shape[1]} features"
    )
    print(
        f"Test set (combined): {test_data.shape[0]} samples, {test_data.shape[1]} features"
    )
    print(
        f"D1 (in-distribution): {d1_data.shape[0]} samples, {d1_data.shape[1]} features"
    )
    print(
        f"D2 (out-of-distribution): {d2_data.shape[0]} samples, {d2_data.shape[1]} features"
    )
    print(
        f"D3 (out-of-distribution): {d3_data.shape[0]} samples, {d3_data.shape[1]} features"
    )

    print("\nGeneration complete!")


def main():
    """Parse command line arguments and generate data."""
    parser = argparse.ArgumentParser(description="Generate synthetic survival data")
    parser.add_argument(
        "--n", type=int, default=1000, help="Number of samples to generate"
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    parser.add_argument(
        "--path", type=str, default="./data", help="Output directory path"
    )

    args = parser.parse_args()

    # Generate data
    generate_data(n=args.n, random_seed=args.seed, file_path=args.path)


if __name__ == "__main__":
    main()
