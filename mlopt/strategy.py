from joblib import Parallel, delayed
import numpy as np
import mlopt.settings as stg
import mlopt.error as e
import mlopt.utils as u
import cvxpy.settings as cps
import scipy.sparse as spa
from time import time
from tqdm import tqdm


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

        # Store hash for comparisons
        self._hash = hash((frozenset(self.tight_constraints),
                           frozenset(self.int_vars)))

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

    def __hash__(self):
        """Overrides default hash implementation"""
        return self._hash

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
    unique = list(set(strategies))

    # Using list
    #  unique = []
    #  # traverse for all elements
    #  for x in strategies:
    #      # check if x exists in unique_list or not
    #      if x not in unique:
    #          unique.append(x)

    return unique


def assign_to_unique_strategy(strategy, unique_strategies):
    y = next((index for (index, s) in enumerate(unique_strategies)
             if strategy == s), -1)
    #  y = -1
    #  n_unique_strategies = len(unique_strategies)
    #  for j in range(n_unique_strategies):
    #      if unique_strategies[j] == strategy:
    #          y = j
    #          break
    if y == -1:
        e.value_error("Strategy not found")
    return y


def encode_strategies(strategies, batch_size=stg.JOBLIB_BATCH_SIZE,
                      parallel=True):
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
    start_time = time()
    unique = unique_strategies(strategies)
    end_time = time()
    stg.logger.info("Extraction time %.3f sec" % (end_time - start_time))
    n_unique_strategies = len(unique)
    stg.logger.info("Found %d unique strategies" % n_unique_strategies)

    # Map strategies to number
    n_jobs = u.get_n_processes() if parallel else 1
    stg.logger.info("Assign samples to unique strategies (n_jobs = %d)"
                    % n_jobs)

    results = Parallel(n_jobs=n_jobs, batch_size=batch_size)(delayed(assign_to_unique_strategy)(s, unique) for s in tqdm(strategies))
    y = np.array(results)

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
