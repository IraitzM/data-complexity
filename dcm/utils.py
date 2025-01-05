import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations


def plot_profile(profile_json):
    sns.set_theme(style="whitegrid")

    # Initialize the matplotlib figure
    _, ax = plt.subplots(figsize=(30, 5))

    # Plot the total crashes
    sns.set_color_codes("pastel")
    sns.barplot(data=profile_json, color="b")

    # Add a legend and informative axis label
    ax.set(ylabel="", xlabel="Complexity metrics")
    sns.despine(left=True, bottom=True)


def ovo(data):
    """
    One-vs-one takes the data in pairs

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
    """
    Max of a column

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    return df.max()


def colMin(df: pd.DataFrame):
    """
    Min of a column

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    return df.min()
