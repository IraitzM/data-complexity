"""
Filename: test_balance.py
"""

import numpy
import unittest
from dcm import NeighborhoodMeasures
from sklearn.datasets import load_iris

"""
Tests
"""


class TestBalance(unittest.TestCase):
    def setUp(self):
        X, y = load_iris(return_X_y=True)

        self.X = X
        self.y = y

    def test_iris(self):
        model = NeighborhoodMeasures()
        model.fit(self.X, self.y)

        numpy.testing.assert_allclose(model.c_N1(), 0.053333, atol=1e-06)
        numpy.testing.assert_allclose(model.c_N3(), 0.053333, atol=1e-06)
        numpy.testing.assert_allclose(model.c_N4(), 0.7, atol=1e-01)
        numpy.testing.assert_allclose(model.c_N5(), 0.106667, atol=1e-06)
        numpy.testing.assert_allclose(model.c_N6(), 0.804356, atol=1e-06)

    def test_iris_100(self):
        model = NeighborhoodMeasures()
        model.fit(self.X[:100], self.y[:100])

        numpy.testing.assert_allclose(model.c_N1(), 0.01, atol=0)
        numpy.testing.assert_allclose(model.c_N3(), 0.0, atol=1e-06)
        numpy.testing.assert_allclose(model.c_N4(), 0.5, atol=1e-01)
        numpy.testing.assert_allclose(model.c_N5(), 0.03, atol=1e-06)
        numpy.testing.assert_allclose(model.c_N6(), 0.5066, atol=1e-06)
