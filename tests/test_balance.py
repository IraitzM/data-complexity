"""
Filename: test_balance.py
"""

import numpy
import unittest
from dcm import BalanceMeasures
from sklearn.datasets import load_iris, load_breast_cancer


class TestBalance(unittest.TestCase):
    def setUp(self):
        X, y = load_iris(return_X_y=True)

        self.X = X
        self.y = y

    def test_iris(self):
        model = BalanceMeasures()
        model.fit(self.X, self.y)

        numpy.testing.assert_allclose(model.c_B1(), 2.220446e-16, atol=0)
        numpy.testing.assert_allclose(model.c_B2(), 0.0, atol=0)

    def test_iris_100(self):
        model = BalanceMeasures()
        model.fit(self.X[:100], self.y[:100])

        numpy.testing.assert_allclose(model.c_B1(), 0.0, atol=0)
        numpy.testing.assert_allclose(model.c_B2(), 0.0, atol=0)

class TestBalance2(unittest.TestCase):
    def setUp(self):
        X, y = load_breast_cancer(return_X_y=True)

        self.X = X
        self.y = y

    def test_cancer(self):
        model = BalanceMeasures()
        model.fit(self.X, self.y)

        numpy.testing.assert_allclose(model.c_B1(), 0.047365, atol=1e-4)
        numpy.testing.assert_allclose(model.c_B2(), 0.12196, atol=1e-4)

    def test_cancer_100(self):
        model = BalanceMeasures()
        model.fit(self.X[:100], self.y[:100])

        numpy.testing.assert_allclose(model.c_B1(), 0.07, atol=1e-2)
        numpy.testing.assert_allclose(model.c_B2(), 0.165138, atol=1e-6)
