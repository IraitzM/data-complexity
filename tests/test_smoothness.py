"""
Filename: test_smoothness.py
"""

import numpy
import unittest
from dcm import ComplexityProfile
from sklearn.datasets import load_iris, load_breast_cancer


class TestSmoothness(unittest.TestCase):
    def setUp(self):
        X, y = load_iris(return_X_y=True)

        self.X = X
        self.y = y

        self.measures = ["S1", "S2", "S3", "S4"]

    def test_iris(self):
        model = ComplexityProfile(self.measures)
        model.fit(self.X, self.y)

        result = model.transform()

        numpy.testing.assert_allclose(result["S1"], 0.036891, atol=1e-4)
        numpy.testing.assert_allclose(result["S2"], 0.524347, atol=1e-4)
        numpy.testing.assert_allclose(result["S3"], 0.031914, atol=1e-4)

    def test_iris_100(self):
        model = ComplexityProfile(self.measures)
        model.fit(self.X[:100], self.y[:100])

        result = model.transform()

        numpy.testing.assert_allclose(result["S1"], 0.006723, atol=1e-4)
        numpy.testing.assert_allclose(result["S2"], 0.523209, atol=1e-4)
        numpy.testing.assert_allclose(result["S3"], 0.0, atol=1e-4)


class TestSmoothness2(unittest.TestCase):
    def setUp(self):
        X, y = load_breast_cancer(return_X_y=True)

        self.X = X
        self.y = y

        self.measures = ["S1", "S2", "S3"]

    def test_cancer(self):
        model = ComplexityProfile(self.measures)
        model.fit(self.X, self.y)

        result = model.transform()

        numpy.testing.assert_allclose(result["S1"], 0.290675, atol=1e-4)
        numpy.testing.assert_allclose(result["S2"], 0.844673, atol=1e-4)
        numpy.testing.assert_allclose(result["S3"], 0.356, atol=1e-4)

    def test_cancer_100(self):
        model = ComplexityProfile(self.measures)
        model.fit(self.X[:100], self.y[:100])

        result = model.transform()

        numpy.testing.assert_allclose(result["S1"], 0.300428, atol=1e-2)
        numpy.testing.assert_allclose(result["S2"], 0.853594, atol=1e-4)
        numpy.testing.assert_allclose(result["S3"], 0.308994, atol=1e-4)
