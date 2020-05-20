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

    def __init(self, x, data):
        """Initialize strategy from problem data."""

        self.tight_constraints = self.tight_constraints(x, data)
        self.int_vars = x[data[cps.INT_IDX]]

    def tight_constraints(self, x, data):
        """Compute tight constraints for solution x

        Args:
            data (TODO): TODO
            x (TODO): TODO

        Returns: TODO

        """
        # Check only inequalities
        F, g = data['F'], data['g']

        tight_constraints = (F.dot(x) - g) <= \
            stg.TIGHT_CONSTRAINTS_TOL * (1 + np.linalg.norm(g, np.inf))

        return tight_constraints

    def __init__(self, tight_constraints=np.array([]), int_vars=np.array([])):

        if not np.array_equal(tight_constraints,
                              tight_constraints.astype(bool)):
            e.value_error("Tight constraints vector is not boolean")

        # Check tight constraints
        #  for _, v in tight_constraints.items():
        #      if np.any(np.logical_or(v < 0, v > 1)):
        #          err = "Tight constraints vector is not boolean."
        #          stg.logger.error(err)
        #          raise ValueError(err)

        # Check that integer variables are non negative
        #  for _, v in int_vars.items():
        #      if np.any(v < 0):
        #          raise ValueError("Integer variables vector " +
        #                           "has negative entries.")

        # Check that tight constraints are not
        self.tight_constraints = tight_constraints
        self.int_vars = int_vars

    #  def _compare_arrays_dict(self, d1, d2):
    #      """Compare dictionaries of numpy arrays"""
    #      if len(d1) != len(d2):
    #          return False
    #      for key in d1.keys():
    #          try:
    #              isequal = np.array_equal(d1[key], d2[key])
    #          except KeyError:
    #              return False
    #          if not isequal:
    #              return False
    #      return True

    #  def __sprint_dict(self, d):
    #      s = ""
    #      for attribute, value in d.items():
    #          s += '      {:>5}: {}\n'.format(attribute, value)
    #      return s.rstrip()

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

            # Compare tight constraints
            same_tight_constraints = \
                np.all(self.tight_constraints == other.tight_constraints)

            if not same_tight_constraints:
                return False

            # Compare integer variables
            same_int_vars = \
                np.all(self.int_vars == other.int_vars)

            return same_tight_constraints and same_int_vars
        else:
            return False

    def parse_data(self, data):
        """TODO: Docstring for parse_data.

        Args:
            data (TODO): TODO

        Returns: TODO

        """
        n_var = data[cps.A].size(1)

        # Edit data by increasing the dimension of A
        # 1. Fix tight constraints: F_active x = g_active
        F_active = data[cps.F][self.tight_constraints]
        g_active = data[cps.G][self.tight_constraints]

        # 2. Fix integer variables: F_fix x = g_fix
        F_fix = spa.eye(n_var)[data[cps.INT_IDX]]
        g_fix = self.int_vars

        # Combine in A and b  # remove (F and g)
        data[cps.A] = spa.vstack([data[cps.A], F_active, F_fix])
        data[cps.B] = np.concatenate([data[cps.B], g_active, g_fix])

        # Remove F and g (no longer have inequalities)
        data[cps.F], data[cps.G] = spa.csr_matrix((0, n_var)), -np.array([])


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
    # Using set
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
