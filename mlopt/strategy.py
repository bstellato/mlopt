# Define strategy
import numpy as np


class Strategy:
    """
    Solving strategy.

    Parameters
    ----------
    int_vars : numpy array
        Value of the integer variables.
    active_constraints : numpy array
        Set of active constraints.
    """

    def __init__(self, int_vars, active_constraints):
        self.int_vars = int_vars
        self.active_constraints = active_constraints

    def __eq__(self, other):
        """Overrides the default equality implementation"""
        if isinstance(other, Strategy):
            same_int_vars = np.array_equal(self.int_vars, other.int_vars)
            same_active_constraints = np.array_equal(self.active_constraints,
                                                     other.active_constraints)
            return same_int_vars and same_active_constraints
        return False


def unique_strategies(strategies):
    """
    Extract unique strategies from array of strategies.

    Parameters
    ----------
    strategies : Strategy array
        Strategies to be processed.

    Returns
    -------
    Strategy array :
        Unique strategies.
    """
    n_int_var = len(strategies[0].int_vars)

    # Construct vector of vectors and get unique elements
    # NB. We need to convert the active constraints from set
    # to numpy arrays to make it work
    strategy_vecs = np.unique(
        np.array(
            [
                np.concatenate((s.int_vars, s.active_constraints))
                for s in strategies
            ],
            dtype=int),
        axis=0,
    )

    # Get unique vectors
    return [Strategy(s[:n_int_var].astype(int), s[n_int_var:].astype(int))
            for s in strategy_vecs]


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
    print("Encoding strategies")
    N = len(strategies)

    print("Getting unique set of strategies")
    unique = unique_strategies(strategies)
    n_unique_strategies = len(unique)
    print("Found %d unique strategies" % n_unique_strategies)

    # Map strategies to number
    y = -1 * np.ones(N, dtype='int')
    for i in range(N):
        for j in range(n_unique_strategies):
            if unique[j] == strategies[i]:
                y[i] = j
                break
        assert y[i] != -1, "Strategy not found"

    return y, unique
