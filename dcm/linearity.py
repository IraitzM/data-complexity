import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator

from .utils import ovo


class LinearityMeasures(BaseEstimator):
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
            method = getattr(self, f"c_{measure}")
            measure_result = method()
            result[measure] = self.summarization(measure_result)
        return result

    def summarization(self, measure):
        if isinstance(measure, (int, float)):
            return measure
        summary_dict = {}
        if "mean" in self.summary:
            summary_dict["mean"] = np.mean(measure)
        if "std" in self.summary:
            summary_dict["std"] = np.std(measure)
        return summary_dict

    @staticmethod
    def ls_featurebased():
        return ["L1"]

    @staticmethod
    def ls_featurebased_multiples():
        return ["L1"]

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

    def c_L1(self):
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

    def c_L2(self):
        raise NotImplementedError

    def c_L3(self):
        raise NotImplementedError
