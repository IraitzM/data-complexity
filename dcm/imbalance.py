import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr


class ImbalanceMeasures(BaseEstimator):
    def __init__(self, measures="all", summary=["mean", "std"]):
        self.measures = measures
        self.summary = summary

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        y = pd.Series(y)

        if isinstance(y, pd.Series) and y.dtype == "object":
            raise ValueError("Label attribute needs to be numeric")

        if X.shape[0] != len(y):
            raise ValueError("X and y must have the same number of rows")

        if self.measures == "all":
            self.measures = self.ls_correlation()

        X.columns = [f"feat_{i}" for i in range(X.shape[1])]

        self.data = pd.concat([self.binarize(X), y.rename("class")], axis=1)

        return self

    def transform(self):
        result = {}
        for measure in self.measures:
            method = getattr(self, f"{measure}")
            measure_result = method()
            result[measure] = self.summarization(measure_result)
        return result

    @staticmethod
    def ls_correlation():
        return ["C1", "C2"]

    @staticmethod
    def ls_correlation_multiples():
        return ["C1", "C2"]

    def summarization(self, measure):
        if isinstance(measure, (int, float)):
            return measure
        summary_dict = {}
        if "mean" in self.summary:
            summary_dict["mean"] = np.mean(measure)
        if "std" in self.summary:
            summary_dict["std"] = np.std(measure)
        return summary_dict

    def binarize(self, X):
        return pd.get_dummies(X, drop_first=True)

    def normalize(self, data):
        scaler = MinMaxScaler()

        if isinstance(data, pd.Series):
            return pd.Series(
                scaler.fit_transform(data.values.reshape(-1, 1)).flatten(),
                index=data.index,
            )
        elif isinstance(data, pd.DataFrame):
            return pd.DataFrame(
                scaler.fit_transform(data), columns=data.columns, index=data.index
            )
        else:
            return scaler.fit_transform(data.reshape(-1, 1))

    def C1(self):
        n = len(self.data["class"])
        class_proportion = self.data["class"].value_counts()

        nc = 0
        proportions = 0.0
        for _, v in class_proportion.items():
            proportions += (v / n) * np.log2(v / n)
            nc += 1

        # Calculate entropy
        return -(1 / np.log2(nc)) * proportions

    def C2(self):
        n = len(self.data["class"])
        class_proportion = self.data["class"].value_counts()

        nc = 0
        sumation = 0.0
        for _, v in class_proportion.items():
            sumation += v / (n - v)
            nc += 1

        IR = (nc - 1) / nc * sumation
        return 1 - 1 / IR
