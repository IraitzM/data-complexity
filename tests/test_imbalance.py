"""
Filename: test_imbalance.py
"""

import numpy
import unittest
from dcm import ImbalanceMeasures
from sklearn.datasets import load_iris, load_breast_cancer


class TestImbalance(unittest.TestCase):
    def setUp(self):
        X, y = load_iris(return_X_y=True)

        self.X = X
        self.y = y

    def test_iris(self):
        model = ImbalanceMeasures()
        model.fit(self.X, self.y)

        numpy.testing.assert_allclose(model.C1(), 1.0, atol=1e-06)
        numpy.testing.assert_allclose(model.C2(), 0.0, atol=1e-06)

    def test_iris_100(self):
        model = ImbalanceMeasures()
        model.fit(self.X[:100], self.y[:100])

        numpy.testing.assert_allclose(model.C1(), 1.0, atol=1e-06)
        numpy.testing.assert_allclose(model.C2(), 0, atol=1e-06)


class TestImbalance2(unittest.TestCase):
    def setUp(self):
        X, y = load_breast_cancer(return_X_y=True)

        self.X = X
        self.y = y

    def test_cancer(self):
        model = ImbalanceMeasures()
        model.fit(self.X, self.y)

        numpy.testing.assert_allclose(model.C1(), 0.952635, atol=1e-06)
        numpy.testing.assert_allclose(model.C2(), 0.12196, atol=1e-06)

    def test_cancer_100(self):
        model = ImbalanceMeasures()
        model.fit(self.X[:100], self.y[:100])

        numpy.testing.assert_allclose(model.C1(), 0.934068, atol=1e-06)
        numpy.testing.assert_allclose(model.C2(), 0.165138, atol=1e-06)
