from ..constants import ACTIVE_CONSTRAINTS_TOL as TOL
import numpy as np


class Solver(object):

    def active_constraints(self, y, eq_idx):
        """
        Get active constraints
        """
        num_constr = len(y)
        active_constr = np.zeros(num_constr, dtype=int)
        # All equality constraints are active
        for i in eq_idx:
            active_constr[i] = 1

        # Check if other constraints are active using dual variables
        for i in range(num_constr):
            if y[i] >= TOL:
                active_constr[i] = 1
            elif y[i] <= -TOL:
                if i not in eq_idx:
                    active_constr[i] = -1

        return active_constr
