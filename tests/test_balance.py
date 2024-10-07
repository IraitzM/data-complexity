"""
Filename: test_balance.py
"""

import numpy
import unittest
from dcm import BalanceMeasures
from sklearn.datasets import load_iris

"""
The TestPassed class iterates through the list of grades. 
For each grade, it checks if it is between 0 and 100. 
If not, it fails the test.
"""


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
