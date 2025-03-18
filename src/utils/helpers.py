from sklearn.model_selection import train_test_split
import torch


def split_dataframe(df, train_size=0.8, val_size=0.2, random_state=None):
    """
    Splits the DataFrame into train, validation, and test sets.

    Parameters:
    df (pd.DataFrame): The DataFrame to split.
    train_size (float): Proportion of the data to be used for the training set.
    val_size (float): Proportion of the data to be used for the validation set.
    random_state (int, optional): Random seed for reproducibility.

    Returns:
    tuple: A tuple containing the train, validation, and test DataFrames.
    """
    assert train_size + val_size == 1, "Proportions must sum to 1."

    # Split the DataFrame into train and remaining sets
    df_train, df_val = train_test_split(
        df, train_size=train_size, random_state=random_state
    )

    return df_train, df_val


def split_df(df, x_label, t_label, e_label, o_label):
    return (
        df[x_label].values,
        df[t_label].values,
        df[e_label].values,
        df[o_label].values,
    )


def predict_cif(method, df, x_label, t_label):
    x_cov = torch.tensor(df[x_label].values, dtype=torch.float32)
    t = torch.tensor(df[t_label].values, dtype=torch.float32)

    cif = method.predict(x_cov, t)

    return cif


def compute_timediff(model, data_loader_val, n_sample, df_type, verbose):
    if df_type == "D1":
        idx_ood = 0.0
    else:
        idx_ood = 1.0

    reg_loss = 0
    for __, (x, t, _, o) in enumerate(data_loader_val):
        argsort_t = torch.argsort(t)
        x_ = x[argsort_t, :].to(model.device)
        o_ = o[argsort_t].to(model.device)

        x_ood = x_[o_ == idx_ood]

        regloss = 0.0
        if x_ood.shape[0] > 0:
            _, regloss_ = model.regularisation(
                x=x_ood, verbose=verbose, n_sample=n_sample
            )
            regloss = regloss_.item()

        reg_loss += regloss

    return reg_loss
