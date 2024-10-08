import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from sklearn.base import BaseEstimator


class NeighborhoodMeasures(BaseEstimator):
    def __init__(self, measures="all", summary=["mean", "std"]):
        self.measures = measures
        self.summary = summary

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        y = pd.Series(y)

        if y.value_counts().min() < 2:
            raise ValueError("Number of examples in the minority class should be >= 2")

        if X.shape[0] != len(y):
            raise ValueError("X and y must have the same number of rows")

        if self.measures == "all":
            self.measures = self.ls_neighborhood()

        X.columns = [f"feat_{i}" for i in range(X.shape[1])]
        self.data = pd.concat([X, y.rename("class")], axis=1)
        self.dst = pdist(X)
        self.dst_matrix = squareform(self.dst)

        return self

    def transform(self):
        result = {}
        for measure in self.measures:
            method = getattr(self, f"{measure}")
            measure_result = method()
            result[measure] = self.summarization(measure_result)
        return result

    @staticmethod
    def ls_neighborhood():
        return ["N1", "N2", "N3", "N4", "N5", "N6"]

    @staticmethod
    def ls_neighborhood_multiples():
        return ["N2", "N3", "N4"]

    def summarization(self, measure):
        if isinstance(measure, (int, float)):
            return measure
        summary_dict = {}
        if "mean" in self.summary:
            summary_dict["mean"] = np.mean(measure)
        if "std" in self.summary:
            summary_dict["std"] = np.std(measure)
        return summary_dict

    def N1(self):
        G = nx.Graph(self.dst_matrix)
        mst = nx.minimum_spanning_tree(G)
        edges = list(mst.edges())
        different_class = sum(
            self.data.iloc[u]["class"] != self.data.iloc[v]["class"] for u, v in edges
        )
        return different_class / self.data.shape[0]

    def intra(self, i):
        same_class = self.data[self.data["class"] == self.data.iloc[i]["class"]].index
        return np.min(self.dst_matrix[i, list(set(same_class) - {i})])

    def inter(self, i):
        diff_class = self.data[self.data["class"] != self.data.iloc[i]["class"]].index
        return np.min(self.dst_matrix[i, diff_class])

    def N2(self):
        intra_distances = np.array([self.intra(i) for i in range(self.data.shape[0])])
        inter_distances = np.array([self.inter(i) for i in range(self.data.shape[0])])
        return 1 - (1 / ((intra_distances / inter_distances) + 1))

    def knn(self, k):
        indices = np.argsort(self.dst_matrix, axis=1)[:, 1 : k + 1]  # exclude self
        return np.array([self.data.iloc[idx]["class"].mode()[0] for idx in indices])

    def N3(self):
        knn_classes = self.knn(2)
        return np.mean(knn_classes != self.data["class"])

    def c_generate(self, n):
        new_data = []
        for _ in range(n):
            sample = self.data.sample(n=1)
            new_instance = sample.iloc[0].copy()
            new_instance["class"] = np.random.choice(self.data["class"].unique())
            new_data.append(new_instance)
        return pd.DataFrame(new_data)

    def N4(self):
        generated_data = self.c_generate(self.data.shape[0])
        combined_data = pd.concat([self.data, generated_data], ignore_index=True)

        combined_dst = pdist(combined_data.drop("class", axis=1))
        combined_dst_matrix = squareform(combined_dst)

        test_dst = combined_dst_matrix[self.data.shape[0] :, : self.data.shape[0]]

        knn_classes = np.array(
            [self.data.iloc[np.argmin(dist)]["class"] for dist in test_dst]
        )
        return np.mean(knn_classes != generated_data["class"])

    def radios(self, i):
        di = self.inter(i)
        j = np.argmin(
            self.dst_matrix[
                i, self.data[self.data["class"] != self.data.iloc[i]["class"]].index
            ]
        )
        dj = self.inter(j)
        k = np.argmin(
            self.dst_matrix[
                j, self.data[self.data["class"] != self.data.iloc[j]["class"]].index
            ]
        )

        if i == k:
            return di / 2
        else:
            return di - self.radios(j)

    def hypersphere(self):
        return np.array([self.radios(i) for i in range(self.data.shape[0])])

    def translate(self, r):
        return self.dst_matrix < r[:, np.newaxis]

    def adherence(self, adh):
        n = []
        h = []
        while adh.shape[0] > 0:
            aux = np.argmax(np.sum(adh, axis=1))
            tmp = np.where(adh[aux])[0]
            dif = np.setdiff1d(np.arange(adh.shape[0]), np.append(tmp, aux))
            adh = adh[dif][:, dif]

            if adh.shape[0] > 0:
                h.append(len(tmp))
            else:
                h.append(1)

            n.append(aux)

        return h, n

    def N5(self):
        r = self.hypersphere()
        adh = self.translate(r)
        h, _ = self.adherence(adh)
        return len(h) / self.data.shape[0]

    def N6(self):
        r = np.array([self.inter(i) for i in range(self.data.shape[0])])
        adh = self.translate(r)
        return 1 - np.sum(adh) / (self.data.shape[0] ** 2)
