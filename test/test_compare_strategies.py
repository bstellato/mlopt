import unittest
import numpy as np
import numpy.testing as npt
from .settings import TEST_TOL as TOL
import mlopt
from mlopt.strategy import Strategy
import os


class TestCompareStrategies(unittest.TestCase):

    def setUp(self):
        # Int variables
        self.int_vars1 = {2: np.array([0, 3, 4]),
                          4: np.array([10, 7, 1])}
        self.int_vars1w = {2: np.array([0, 3, 4]),
                           4: np.array([-1, 10, 7, 1])}
        self.int_vars2 = {7: np.array([3, 8, 10]),
                          15: np.array([0, 7, 9])}
        self.int_vars3 = {27: np.array([0, 7, 9]),
                          15: np.array([0, 7, 9]),
                          10: np.array([3, 8, 2])
        self.int_vars3a = {27: np.array([0, 7, 9]),
                           16: np.array([0, 7, 9]),
                           10: np.array([3, 8, 2])

        # Binding constraints
        self.binding_cons1 = {1: np.array([0, 1, 0, 2]),
                              2: np.array([0, 7, 2, 0])}


        # TODO: Continue defining setup
        # Add tests for the next functions

    def test_same_strategy(self):
        """Test same strategy"""
        pass

    def test_wrong_values_strategy(self):
        """Test strategy with wrong values"""
        self.assertRaises(ValueError,
                          Strategy(self.int_vars1w,
                                   self.binding_cons1))

    def test_unique_filter(self):
        """Test unique strategies filter"""
        pass
