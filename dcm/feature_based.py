import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder

from .utils import ovo


class FeatureBasedMeasures(BaseEstimator):
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
        self.data = None

    def fit(self, X, y):
        """_summary_

        Args:
            X (_type_): _description_
            y (_type_): _description_

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        y = pd.Series(y)

        if y.value_counts().min() < 2:
            raise ValueError("Number of examples in the minority class should be >= 2")

        if X.shape[0] != len(y):
            raise ValueError("X and y must have the same number of rows")

        if self.measures == "all":
            self.measures = self.ls_featurebased()

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
    def ls_featurebased():
        return ["F1", "F1v", "F2", "F3", "F4"]

    @staticmethod
    def ls_featurebased_multiples():
        return ["F1", "F1v", "F2", "F3", "F4"]

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
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns
        if not categorical_cols.empty:
            enc = OneHotEncoder(handle_unknown="ignore")
            encoded = enc.fit_transform(X[categorical_cols])
            encoded_df = pd.DataFrame(
                encoded, columns=enc.get_feature_names_out(categorical_cols)
            )
            X = pd.concat([X.drop(columns=categorical_cols), encoded_df], axis=1)
        return X

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
        TODO: validate
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
