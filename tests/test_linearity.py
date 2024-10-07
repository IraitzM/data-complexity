"""
Filename: test_linearity.py
"""

import numpy
import unittest
from dcm import LinearityMeasures
from sklearn.datasets import load_iris


class TestBalance(unittest.TestCase):
    def setUp(self):
        X, y = load_iris(return_X_y=True)

        self.X = X
        self.y = y

    def test_iris(self):
        model = LinearityMeasures()
        model.fit(self.X, self.y)

        numpy.testing.assert_allclose(model.c_L1(), 0.01, atol=1e-2)

    def test_iris_100(self):
        model = LinearityMeasures()
        model.fit(self.X[:100], self.y[:100])

        numpy.testing.assert_allclose(model.c_L1(), 0.0, atol=0)
