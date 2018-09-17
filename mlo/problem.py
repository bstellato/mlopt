# Define optimization problem
import numpy as np
from . import constants as con


class ProblemData:
    def __init__(self, c, l, A, u, int_idx=np.array([])):
        self.c = c
        self.A = A
        self.l = l
        self.u = u
        self.int_idx = int_idx

    def eq_ineq(self):
        """
        Return equality and inequality constraints
        """
        n_con = len(self.l)
        eq = np.where(self.u - self.l) <= con.TOL
        ineq = np.array(set(range(n_con)) - set(eq))
        return eq, ineq


class OptimizationProblem:
    def is_mip(self):
        return len(self.data) > 0

    def cost(self, x):
        return np.dot(self.data.c, x)

