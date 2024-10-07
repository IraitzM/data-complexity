import pandas as pd
from itertools import combinations


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
