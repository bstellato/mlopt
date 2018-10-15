import unittest
import numpy as np
from mlopt.strategy import Strategy, encode_strategies


class TestCompareStrategies(unittest.TestCase):

    def setUp(self):
        # Int variables
        self.int_vars1 = {2: np.array([0, 3, 4]),
                          4: np.array([10, 7, 1])}
        self.int_vars1cp = {2: np.array([0, 3, 4]),
                            4: np.array([10, 7, 1])}
        self.int_vars1w = {2: np.array([0, 3, 4]),
                           4: np.array([-1, 10, 7, 1])}
        self.int_vars2 = {7: np.array([3, 8, 10]),
                          15: np.array([0, 7, 9])}
        self.int_vars3 = {27: np.array([0, 7, 9]),
                          15: np.array([0, 7, 9]),
                          10: np.array([3, 8, 2])}
        self.int_vars3a = {27: np.array([0, 7, 9]),
                           16: np.array([0, 7, 9]),
                           10: np.array([3, 8, 2])}

        # Binding constraints
        self.binding_cons1 = {1: np.array([0, 1, 0, 1]),
                              2: np.array([0, 1, 1, 0])}
        self.binding_cons1cp = {1: np.array([0, 1, 0, 1]),
                                2: np.array([0, 1, 1, 0])}
        self.binding_cons1w = {1: np.array([0, 1, 0, 1]),
                               2: np.array([-1, 1, 1, 0])}
        self.binding_cons2 = {6: np.array([0, 1, 0, 1]),
                              19: np.array([1, 1, 0, 0, 0])}
        self.binding_cons3 = {6: np.array([0, 1, 0, 1]),
                              8: np.array([1, 1]),
                              19: np.array([1, 1, 0, 0, 0])}

    def test_same_strategy1(self):
        """Test same strategy"""
        s1 = Strategy(self.binding_cons3, self.int_vars3)
        s2 = Strategy(self.binding_cons3, self.int_vars3)
        s3 = Strategy(self.binding_cons1, self.int_vars2)

        self.assertTrue(s1 == s2)
        self.assertTrue(s2 != s3)

    def test_same_strategy2(self):
        """Test same strategy"""
        s1 = Strategy(self.binding_cons1, self.int_vars1)
        s2 = Strategy(self.binding_cons1cp, self.int_vars1cp)
        self.assertTrue(s1 == s2)

    def test_wrong_values_strategy(self):
        """Test strategy with wrong values"""
        self.assertRaises(ValueError,
                          Strategy, self.binding_cons1,
                          self.int_vars1w)
        self.assertRaises(ValueError,
                          Strategy, self.binding_cons1w,
                          self.int_vars1)

    def test_unique_filter(self):
        """Test unique strategies filter"""

        s1 = Strategy(self.binding_cons1, self.int_vars1)
        s2 = Strategy(self.binding_cons3, self.int_vars3)
        s3 = Strategy(self.binding_cons1cp, self.int_vars1cp)
        s4 = Strategy(self.binding_cons3, self.int_vars3)
        s5 = Strategy(self.binding_cons1cp, self.int_vars1cp)
        s6 = Strategy(self.binding_cons1, self.int_vars2)

        y, unique = encode_strategies([s1, s2, s3, s4, s5, s6])

        self.assertTrue(np.array_equal(y, np.array([0, 1, 0, 1, 0, 2])))
        self.assertTrue(unique == [s1, s2, s6])
