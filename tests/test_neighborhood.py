"""
Filename: test_neighborhood.py
"""

import numpy
import unittest
from dcm import NeighborhoodMeasures
from sklearn.datasets import load_iris, load_breast_cancer


class TestNeighborhood(unittest.TestCase):
    def setUp(self):
        X, y = load_iris(return_X_y=True)

        self.X = X
        self.y = y

    def test_iris(self):
        model = NeighborhoodMeasures()
        model.fit(self.X, self.y)

        numpy.testing.assert_allclose(model.N1(), 0.053333, atol=1e-06)
        numpy.testing.assert_allclose(model.N3(), 0.053333, atol=1e-06)
        numpy.testing.assert_allclose(model.N4(), 0.7, atol=1e-01)
        numpy.testing.assert_allclose(model.N5(), 0.106667, atol=1e-06)
        numpy.testing.assert_allclose(model.N6(), 0.804356, atol=1e-06)

    def test_iris_100(self):
        model = NeighborhoodMeasures()
        model.fit(self.X[:100], self.y[:100])

        numpy.testing.assert_allclose(model.N1(), 0.01, atol=0)
        numpy.testing.assert_allclose(model.N3(), 0.0, atol=1e-06)
        numpy.testing.assert_allclose(model.N4(), 0.5, atol=1e-01)
        numpy.testing.assert_allclose(model.N5(), 0.03, atol=1e-06)
        numpy.testing.assert_allclose(model.N6(), 0.5066, atol=1e-06)

class TestNeighborhood2(unittest.TestCase):
    def setUp(self):
        X, y = load_breast_cancer(return_X_y=True)

        self.X = X
        self.y = y

    def test_cancer(self):
        model = NeighborhoodMeasures()
        model.fit(self.X, self.y)

        numpy.testing.assert_allclose(model.N1(), 0.086116, atol=1e-06)
        numpy.testing.assert_allclose(model.N3(), 0.091388, atol=1e-06)
        numpy.testing.assert_allclose(model.N4(), 0.534271, atol=1e-01)
        numpy.testing.assert_allclose(model.N5(), 0.003515, atol=1e-06)
        numpy.testing.assert_allclose(model.N6(), 0.912293, atol=1e-06)

    def test_cancer_100(self):
        model = NeighborhoodMeasures()
        model.fit(self.X[:100], self.y[:100])

        numpy.testing.assert_allclose(model.N1(), 0.1, atol=0)
        numpy.testing.assert_allclose(model.N3(), 0.08, atol=1e-06)
        numpy.testing.assert_allclose(model.N4(), 0.5, atol=1e-01)
        numpy.testing.assert_allclose(model.N5(), 0.02, atol=1e-06)
        numpy.testing.assert_allclose(model.N6(), 0.7953, atol=1e-06)
