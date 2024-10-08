import pandas as pd
import numpy as np
import networkx as nx
from sklearn.base import BaseEstimator
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KNeighborsRegressor

class Smoothness(BaseEstimator):
    def __init__(self, measures="all", summary=["mean", "std"]):
        """
        Class initialization

        Args:
            measures (str, optional): _description_. Defaults to "all".
            summary (list, optional): _description_. Defaults to ["mean", "std"].
        """
        self.measures = measures
        self.summary = summary

        # Initialized
        self.x = None
        self.y = None
        self.d = None

    def fit(self, x, y):
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)

        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]

        if isinstance(y, pd.Categorical):
            raise ValueError("label attribute needs to be numeric")

        if len(x) != len(y):
            raise ValueError("x and y must have same number of rows")

        if self.measures == "all":
            self.measures = self.ls_smoothness()
        else:
            self.measures = [m for m in self.measures if m in self.ls_smoothness()]

        x.columns = [f"col_{i}" for i in range(len(x.columns))]
        x = self.normalize(x)
        y = self.normalize(pd.DataFrame(y)).iloc[:, 0]

        sorted_indices = np.argsort(y)
        self.x = x.iloc[sorted_indices].reset_index(drop=True)
        self.y = y.iloc[sorted_indices].reset_index(drop=True)

        self.d = squareform(pdist(x))

    def transform(self):
        result = {}
        for measure in self.measures:
            method = getattr(self, f"{measure}")
            measure_result = method()
            result[measure] = self.summarization(measure_result)
        return result

    @staticmethod
    def ls_smoothness():
        return ["S1", "S2", "S3", "S4"]

    def S1(self):
        g = nx.from_numpy_array(self.d)
        tree = nx.minimum_spanning_tree(g)
        edges = list(tree.edges())
        aux = np.abs(
            self.y.iloc[[e[0] for e in edges]].values - self.y.iloc[[e[1] for e in edges]].values
        )
        return aux / (aux + 1)

    def S2(self):
        pred = self.d[range(len(self.d) - 1), range(1, len(self.d))]
        return pred / (pred + 1)


    def S3(self):
        np.fill_diagonal(self.d, np.inf)
        pred = self.y.iloc[np.argmin(self.d, axis=1)].values
        aux = (pred - self.y.values) ** 2
        return aux / (aux + 1)

    def normalize(self, df):
        return (df - df.mean()) / df.std()

    def summarization(self, measure):
        if isinstance(measure, (int, float)):
            return measure

        summary_dict = {}
        if "mean" in self.summary:
            summary_dict["mean"] = np.mean(measure)
        if "std" in self.summary:
            summary_dict["std"] = np.std(measure)
        return summary_dict

"""
    def S4(self):
        test = r_generate(x, y, len(x))
        knn = KNeighborsRegressor(n_neighbors=1)
        knn.fit(x, y)
        pred = knn.predict(test.iloc[:, :-1])
        aux = (pred - test.iloc[:, -1].values) ** 2
        return aux / (aux + 1)

    def summarization(self, measure):
        if "return" in self.summary:
            return measure
        result = {}
        if "mean" in self.summary:
            result["mean"] = np.mean(measure)
        if "sd" in self.summary:
            result["sd"] = np.std(measure)
        return result


    def r_interpolation(x, y, i):
        aux = x.iloc[(i - 1) : i + 1].copy()
        rnd = np.random.uniform()
        for j in range(len(x.columns)):
            if np.issubdtype(x.iloc[:, j].dtype, np.number):
                aux.iloc[0, j] = aux.iloc[0, j] + (aux.iloc[1, j] - aux.iloc[0, j]) * rnd
            else:
                aux.iloc[0, j] = np.random.choice(aux.iloc[:, j])

        tmp = y.iloc[(i - 1) : i + 1].copy()
        rnd = np.random.uniform()
        tmp.iloc[0] = tmp.iloc[0] * rnd + tmp.iloc[1] * (1 - rnd)

        return pd.concat([aux.iloc[0], pd.Series([tmp.iloc[0]], index=["y"])])


    def r_generate(x, y, n):
        tmp = pd.DataFrame([r_interpolation(x, y, i) for i in range(1, n)])
        tmp.columns = list(x.columns) + ["y"]
        return tmp
"""