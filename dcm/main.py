import pandas as pd
import numpy as np
import networkx as nx
from .utils import ovo

from scipy.stats import entropy
from sklearn.base import BaseEstimator
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KNeighborsRegressor

class ComplexityProfile(BaseEstimator):
    def __init__(self, measures="all"):
        """
        Class initialization

        Args:
            measures (str, optional): _description_. Defaults to "all".
            summary (list, optional): _description_. Defaults to ["mean", "std"].
        """
        self.measures = measures

        # Initialized
        self.x = None
        self.y = None
        self.d = None

    def normalize(self, df):
        return (df - df.mean()) / df.std()

    def binarize(self, X):
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns
        if not categorical_cols.empty:
            enc = OneHotEncoder(handle_unknown="ignore")
            encoded = enc.fit_transform(X[categorical_cols])
            encoded_df = pd.DataFrame(
                encoded, columns=enc.get_feature_names_out(categorical_cols)
            )
            X = pd.concat([X.drop(columns=categorical_cols), encoded_df], axis=1)
        return X

    def fit(self, x, y):
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)

        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]

        y = pd.Series(y)
        if y.value_counts().min() < 2:
            raise ValueError("Number of examples in the minority class should be >= 2")

        if isinstance(y, pd.Categorical):
            raise ValueError("label attribute needs to be numeric")

        if len(x) != len(y):
            raise ValueError("x and y must have same number of rows")

        if self.measures == "all":
            self.measures = self.ls_measures()
        else:
            self.measures = [m for m in self.measures if m in self.ls_measures()]

        x.columns = [f"feat_{i}" for i in range(x.shape[1])]

        self.data = pd.concat([self.binarize(x), y.rename("class")], axis=1)
        self.dst = pdist(x)
        self.dst_matrix = squareform(self.dst)

        x = self.normalize(x)
        y = self.normalize(pd.DataFrame(y)).iloc[:, 0]

        sorted_indices = np.argsort(y)
        self.x = x.iloc[sorted_indices].reset_index(drop=True)
        self.y = y.iloc[sorted_indices].reset_index(drop=True)

        self.d = squareform(pdist(x))

    def summarization(self, measure):
        return float(np.mean(measure))

    def transform(self, return_type:str = "dict"):
        result = {}
        for measure in self.measures:
            method = getattr(self, f"{measure}")
            measure_result = method()
            result[measure] = self.summarization(measure_result)

        if return_type == "df":
            return pd.DataFrame.from_records([result])

        return result

    @staticmethod
    def ls_measures():
        return [
            # Feature based
            "F1", "F1v", "F2", "F3",
            # Linearity
            "L1",
            # Neighborhood
            "N1", "N2", "N3", "N4", #"N5", "N6",
            #Balance
            "B1", "B2",
            # Smoothness
            "S1", "S2", "S3", "S4",
            # Correlation
            "C1", "C2",
            # Dimensionality
            "T2", "T3", "T4"
            ]

    # Feature based

    def branch(self, j):
        return self.data[self.data["class"] == j].drop("class", axis=1)

    def F1(self):
        """
        Maximum Fisher's Discriminant Ratio (F1)

        Returns:
            float: F1
        """
        X = self.data.drop("class", axis=1)
        overall_mean = X.mean()

        # Numerator
        numerator = sum(
            len(self.branch(clss)) * (self.branch(clss).mean() - overall_mean) ** 2
            for clss in self.data["class"].unique()
        )

        # Denominator
        denominator = sum(
            ((self.branch(clss) - self.branch(clss).mean()) ** 2).sum()
            for clss in self.data["class"].unique()
        )

        # Get max of all fi
        max_ri = 0.0
        for n, d in zip(numerator, denominator):
            if d == 0.0:
                max_ri = np.inf
            elif n / d > max_ri:
                max_ri = n / d

        return 1 / (max_ri + 1)

    def F1v(self):
        """
        The Directional-vector Maximum Fisher's Discriminant Ratio.

        Uses one-vs-one for multiclass problems.
        """

        def dvector(data):
            classes = data["class"].unique()
            a = self.branch(classes[0])
            b = self.branch(classes[1])

            c1 = a.mean()
            c2 = b.mean()

            W = (len(a) / len(data)) * a.cov() + (len(b) / len(data)) * b.cov()
            B = np.outer(c1 - c2, c1 - c2)
            d = np.linalg.pinv(W) @ (c1 - c2)

            return (d.T @ B @ d) / (d.T @ W @ d)

        ovo_data = ovo(self.data)
        f1v = [dvector(data) for data in ovo_data]
        return 1 / (np.array(f1v) + 1)

    def F2(self):
        """
        Value of overlapping region
        """

        def region_over(data):
            classes = data["class"].unique()
            a = self.branch(classes[0])
            b = self.branch(classes[1])

            maxmax = np.maximum(a.max(), b.max())
            minmin = np.minimum(a.min(), b.min())

            over = np.maximum(
                np.minimum(maxmax, b.max()) - np.maximum(minmin, a.min()), 0
            )
            rang = maxmax - minmin
            return np.prod(over / rang)

        ovo_data = ovo(self.data)
        return [region_over(data) for data in ovo_data]

    def F3(self):
        def non_overlap(data):
            classes = data["class"].unique()
            a = self.branch(classes[0])
            b = self.branch(classes[1])

            minmax = np.minimum(a.max(), b.max())
            maxmin = np.maximum(a.min(), b.min())

            return (
                (data.drop("class", axis=1) < maxmin)
                | (data.drop("class", axis=1) > minmax)
            ).sum() / len(data)

        ovo_data = ovo(self.data)
        f3 = [non_overlap(data) for data in ovo_data]
        return 1 - np.max(f3, axis=0)

    def F4(self):
        def removing(data):
            while True:
                non_overlap = (
                    (data.drop("class", axis=1) < data.drop("class", axis=1).min())
                    | (data.drop("class", axis=1) > data.drop("class", axis=1).max())
                ).sum()
                col = non_overlap.idxmax()
                data = data[data[col] == False].drop(col, axis=1)

                if (
                    len(data) == 0
                    or len(data.columns) == 1
                    or len(data["class"].unique()) == 1
                ):
                    break

            return data

        ovo_data = ovo(self.data)
        return [len(removing(data)) / len(data) for data in ovo_data]

    # Smoothness

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

    def S4(self):
        test = self.r_generate()
        knn = KNeighborsRegressor(n_neighbors=1)
        knn.fit(self.x, self.y)
        pred = knn.predict(test.iloc[:, :-1])
        aux = (pred - test.iloc[:, -1].values) ** 2
        return aux / (aux + 1)

    def r_interpolation(self, i):
        aux = self.x.iloc[(i - 1) : i + 1].copy()
        rnd = np.random.uniform()
        for j in range(len(self.x.columns)):
            if np.issubdtype(self.x.iloc[:, j].dtype, np.number):
                aux.iloc[0, j] = aux.iloc[0, j] + (aux.iloc[1, j] - aux.iloc[0, j]) * rnd
            else:
                aux.iloc[0, j] = np.random.choice(aux.iloc[:, j])

        tmp = self.y.iloc[(i - 1) : i + 1].copy()
        rnd = np.random.uniform()
        tmp.iloc[0] = tmp.iloc[0] * rnd + tmp.iloc[1] * (1 - rnd)

        return pd.concat([aux.iloc[0], pd.Series([tmp.iloc[0]], index=["y"])])

    def r_generate(self):
        n = len(self.x)

        tmp = pd.DataFrame([self.r_interpolation(i) for i in range(1, n)])
        tmp.columns = list(self.x.columns) + ["y"]
        return tmp

    # Correlation
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

    # Linearity
    def L1(self):
        """
        Sum of error distance by linear programming

        Returns:
            float: L1
        """
        X = self.data.drop("class", axis=1)
        ovo_data = ovo(self.data)

        error_dist = []
        for data in ovo_data:
            X = data.drop("class", axis=1)
            y = data["class"]

            clf = svm.SVC(kernel="linear")
            clf.fit(X, y)

            error_dist.append(sum(clf.predict(X) != y) / len(y))

        return 1 - (1 / (1 + sum(error_dist)))

    def L2(self):
        raise NotImplementedError

    def L3(self):
        raise NotImplementedError

    # Neighborhood
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

    def T1(self):
        raise NotImplementedError

    def LSC(self):
        raise NotImplementedError

    # Balance
    def B1(self):
        """
        Class balance

        Returns:
            float: Value of the entropy associated with the label
        """
        c = -1 / np.log(self.y.nunique())
        i = self.y.value_counts(normalize=True)
        return 1 + c * entropy(i)

    def B2(self):
        ii = self.y.value_counts()
        nc = len(ii)
        aux = ((nc - 1) / nc) * np.sum(ii / (len(self.y) - ii))
        return 1 - (1 / aux)

    # Dimension
    def pca_variance(self):
        """Python equivalent of R's pca function."""
        pca = PCA()
        pca.fit(self.x)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        # Find number of components needed for 95% variance
        n_components = np.argmax(cumsum >= 0.95) + 1
        return n_components

    def T2(self):
        """Ratio of number of features to number of instances."""
        return self.x.shape[1] / self.x.shape[0]

    def T3(self):
        """Ratio of PCA components to number of instances."""
        return self.pca_variance() / self.x.shape[0]

    def T4(self):
        """Ratio of PCA components to number of features."""
        return self.pca_variance() / self.x.shape[1]
