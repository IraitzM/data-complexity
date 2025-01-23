from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder


def plot_profile(profile_json):
    """Plots the barplot with the complexity measures

    Args:
        profile_json (_type_): _description_
    """
    sns.set_theme(style="whitegrid")

    # Initialize the matplotlib figure
    _, ax = plt.subplots(figsize=(30, 5))

    # Plot the total crashes
    sns.set_color_codes("pastel")
    sns.barplot(data=profile_json, color="b")

    # Add a legend and informative axis label
    ax.set(ylabel="", xlabel="Complexity metrics")
    ax.set(ylim=(0.0, 1.0))
    sns.despine(left=True, bottom=True)


def ovo(data):
    """One-vs-one takes the data in pairs

    Args:
        data (DataFrame): Data with class column informed

    Returns:
        list: Binary list indexing the two sub-groups
    """
    return [
        data[data["class"].isin(combo)]
        for combo in combinations(data["class"].unique(), 2)
    ]


def colMax(df: pd.DataFrame):
    """Max of a column

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    return df.max()


def colMin(df: pd.DataFrame):
    """Min of a column

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    return df.min()


def normalize(df: pd.DataFrame):
    """Normalization of data

    Args:
        df (pd.DataFrame): Dataframe, all numeric.

    Returns:
        pd.DataFrame: Normalizar dataset
    """
    return (df - df.mean()) / df.std()


def binarize(X):
    """_summary_

    Args:
        X (_type_): _description_

    Returns:
        _type_: _description_
    """
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    if not categorical_cols.empty:
        enc = OneHotEncoder(handle_unknown="ignore")
        encoded = enc.fit_transform(X[categorical_cols])
        encoded_df = pd.DataFrame(
            encoded, columns=enc.get_feature_names_out(categorical_cols)
        )
        X = pd.concat([X.drop(columns=categorical_cols), encoded_df], axis=1)
    return X
