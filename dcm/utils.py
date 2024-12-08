import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from itertools import combinations

def gen_datasets(n_samples:int = 300, trials:int = 1, proportion:float = 0.5)->list:
    """Generate desired daatsets

    Args:
        n_samples (int, optional): _description_. Defaults to 300.
        trials (int, optional): _description_. Defaults to 1.
        proportion (float, optional): _description_. Defaults to 0.5.
    """
    y = np.random.binomial(trials, proportion, n_samples)

    x1 = np.random.random(n_samples)
    x2 = 0.5+(-1+y)*0.20 # x2 sets 0 samples on 0.20 value and 1 on 0.70
    easy = np.concatenate((x1[:, None], x2[:, None]), axis=1)

    x2 = 0.75+y*-1*np.random.normal(0.5, 0.1, n_samples)
    still_easy = np.concatenate((x1[:, None], x2[:, None]), axis=1)

    X_blob, y_blob = make_blobs(n_samples=[n_samples, int(n_samples*proportion)], n_features=2,random_state=0)

    x2 = x1+(y-1)*0.2
    linear = np.concatenate((x1[:, None], x2[:, None]), axis=1)

    x1 = np.random.random(n_samples)
    x2 = np.random.random(n_samples)
    random = np.concatenate((x1[:, None], x2[:, None]), axis=1)

    return {"X" : easy, "y" : y},{"X" : still_easy, "y" : y}, {"X" : X_blob, "y" : y_blob}, {"X" : linear, "y" : y}, {"X" : random, "y" : y}

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
