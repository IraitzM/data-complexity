"""
Filename: test_features.py
"""

import numpy
import unittest
from dcm import FeatureBasedMeasures
from sklearn.datasets import load_iris, load_breast_cancer


class TestFeatures(unittest.TestCase):
    def setUp(self):
        X, y = load_iris(return_X_y=True)

        self.X = X
        self.y = y

    def test_iris(self):
        model = FeatureBasedMeasures()
        model.fit(self.X, self.y)

        numpy.testing.assert_allclose(model.F1(), 0.058628, atol=1e-06)
        numpy.testing.assert_allclose(model.F4(), 0.0, atol=1e-06)

    def test_iris_100(self):
        model = FeatureBasedMeasures()
        model.fit(self.X[:100], self.y[:100])

        numpy.testing.assert_allclose(model.F1(), 0.059119, atol=1e-06)
        numpy.testing.assert_allclose(model.F4(), 0.0, atol=1e-06)

class TestFeatures2(unittest.TestCase):
    def setUp(self):
        X, y = load_breast_cancer(return_X_y=True)

        self.X = X
        self.y = y

    def test_cancer(self):
        model = FeatureBasedMeasures()
        model.fit(self.X, self.y)

        numpy.testing.assert_allclose(model.F1(), 0.370253, atol=1e-06)
        numpy.testing.assert_allclose(model.F4(), 0.0, atol=1e-06)

    def test_cancer_100(self):
        model = FeatureBasedMeasures()
        model.fit(self.X[:100], self.y[:100])

        numpy.testing.assert_allclose(model.F1(), 0.399597, atol=1e-06)
        numpy.testing.assert_allclose(model.F4(), 0.0, atol=1e-06)
