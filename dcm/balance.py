import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from scipy.stats import entropy


class BalanceMeasures(BaseEstimator):
    def __init__(self, measures="all"):
        self.measures = measures

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        y = pd.Series(y)

        if X.shape[0] != len(y):
            raise ValueError("X and y must have the same number of rows")

        if self.measures == "all":
            self.measures = self.ls_balance()

        self.y = y

        return self

    def transform(self):
        result = {}
        for measure in self.measures:
            method = getattr(self, f"{measure}")
            result[measure] = method()
        return result

    @staticmethod
    def ls_balance():
        return ["B1", "B2"]

    def B1(self):
        c = -1 / np.log(self.y.nunique())
        i = self.y.value_counts(normalize=True)
        return 1 + c * entropy(i)

    def B2(self):
        ii = self.y.value_counts()
        nc = len(ii)
        aux = ((nc - 1) / nc) * np.sum(ii / (len(self.y) - ii))
        return 1 - (1 / aux)
