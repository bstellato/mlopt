import numpy as np
import mlopt.settings as stg
import mlopt.error as e
import cvxpy.settings as cps
import scipy.sparse as spa


class Strategy(object):
    """
    Solving strategy.

    Parameters
    ----------
    tight_constraints : numpy bool array
        Set of tight constraints. The values are numpy bool arrays
        (True/False for tight/non tight).
    int_vars : numpy bool array
        Value of the integer variables. The values are numpy int arrays.
    """

    def __init__(self, x, data):
        """Initialize strategy from problem data."""

        self.tight_constraints = self.get_tight_constraints(x, data)
        self.int_vars = x[data[cps.INT_IDX]]

    def get_tight_constraints(self, x, data):
        """Compute tight constraints for solution x

        Args:
            data (TODO): TODO
            x (TODO): TODO

        Returns: TODO

        """
        # Check only inequalities
        F, g = data[cps.F], data[cps.G]

        tight_constraints = np.array([], dtype=np.bool)

        # Constraint is tight if ||F * x - g|| <= eps (1 + rel_tol)
        if F.size > 0:
            tight_constraints = np.abs(F.dot(x) - g) <= \
                stg.TIGHT_CONSTRAINTS_TOL * (1 + np.linalg.norm(g, np.inf))

        return tight_constraints

    def __repr__(self):
        string = "Strategy\n"
        string += "  - Tight constraints:\n"
        string += self.tight_constraints.__str__() + "\n"
        if len(self.int_vars) > 0:
            string += "  - Integer variables values:\n"
            string += self.int_vars.__str__() + "\n"
        return string

    # TODO: Implement hash if unique list comparison starts getting slow
    #  def __hash__(self):
    #      """Overrides default hash implementation"""
    #      f_int_vars = frozenset(self.int_vars)
    #      f_tight_constraints = frozenset(self.tight_constraints)
    #      return hash((f_tight_constraints, f_int_vars))

    def __eq__(self, other):
        """Overrides the default equality implementation"""
        if isinstance(other, Strategy):

            if np.any(self.tight_constraints != other.tight_constraints):
                return False

            if np.any(self.int_vars != other.int_vars):
                return False

            return True
        else:
            return False

    def accepts(self, data):
        """Check if strategy is compatible with current problem.
        If not, it raises an error.

        TODO: Add check to see if we match problem type

        Args:
            data (TODO): TODO

        """

        if len(self.tight_constraints) != data['n_ineq']:
            e.warn("Tight constraints not compatible with problem. " +
                   "Different than the number of inequality constraints.")
            return False

        if len(self.int_vars) != len(data['int_vars_idx']):
            e.warn("Integer variables not compatible " +
                   "with problem. IDs not " +
                   "matching an integer variable.")
            return False

        return True

    def apply(self, data, inverse_data):
        """TODO: Docstring for apply.

        Args:
            data (TODO): TODO
            inverse_data (TODO): TODO

        Returns: TODO

        """
        n_eq, n_var = data[cps.A].shape
        n_ineq = data[cps.F].shape[0]

        # Edit data by increasing the dimension of A
        # 1. Fix tight constraints: F_active x = g_active
        A_active = data[cps.F][self.tight_constraints]
        b_active = data[cps.G][self.tight_constraints]

        # 2. Fix integer variables: F_fix x = g_fix
        A_fix = spa.eye(n_var, format='csc')[data[cps.INT_IDX]]
        b_fix = self.int_vars

        # Combine in A_ref and b_red
        data[cps.A + "_red"] = spa.vstack([data[cps.A], A_active, A_fix])
        data[cps.B + "_red"] = np.concatenate([data[cps.B], b_active, b_fix])

        # Store inverse data
        inverse_data['tight_constraints'] = self.tight_constraints
        inverse_data['int_vars'] = self.int_vars
        inverse_data['n_eq'] = n_eq
        inverse_data['n_ineq'] = n_ineq


def unique_strategies(strategies):
    """
    Extract unique strategies from array of strategies.

    Parameters
    ----------
    strategies : Strategy list
        Strategies to be processed.

    Returns
    -------
    Strategy set :
        Unique strategies.
    """
    # Using set (we must define hash to use this)
    #  unique = list(set(strategies))

    # Using list
    unique = []
    # traverse for all elements
    for x in strategies:
        # check if x exists in unique_list or not
        if x not in unique:
            unique.append(x)

    return unique


def encode_strategies(strategies):
    """
    Encode strategies


    Parameters
    ----------
    strategies : Strategies array
        Array of strategies to be encoded.

    Returns
    -------
    numpy array
        Encodings for each strategy in strategies.
    Strategies array
        Array of unique strategies.
    """
    stg.logger.info("Encoding strategies")
    N = len(strategies)

    stg.logger.info("Getting unique set of strategies")
    unique = unique_strategies(strategies)
    n_unique_strategies = len(unique)
    stg.logger.info("Found %d unique strategies" % n_unique_strategies)

    # Map strategies to number
    y = -1 * np.ones(N, dtype='int')
    for i in range(N):
        for j in range(n_unique_strategies):
            if unique[j] == strategies[i]:
                y[i] = j
                break
        assert y[i] != -1, "Strategy not found"

    return y, unique


def strategy2array(s):
    """Convert strategy to array"""
    return np.concatenate([s.tight_constraints, s.int_vars])


def strategy_distance(a, b):
    """Compute manhattan distance between strategy a and b."""
    # Convert strategies to array
    a_array = strategy2array(a)
    b_array = strategy2array(b)

    return np.linalg.norm(a_array - b_array, 1) / len(a_array)
