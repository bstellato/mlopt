# Define strategy
import numpy as np


class Strategy(object):
    """
    Solving strategy.

    Parameters
    ----------
    active_constraints : dict of numpy int arrays
        Set of active constraints. The keys are the CVXPY constraint id.
        The values are numpy int arrays (1/0 for active/inactive).
    int_vars : dict of numpy int arrays
        Value of the integer variables. The keys are CVXPY variable id.
        The values are numpy int arrays.
    """

    def __init__(self, active_constraints, int_vars):
        self.active_constraints = active_constraints
        self.int_vars = int_vars

    def _compare_arrays_dict(self, d1, d2):
        """Compare dictionaries of numpy arrays"""
        for key in d1.keys():
            if not np.array_equal(d1[key], d2[key]):
                return False
        return True

    def __sprint_dict(self, d):
        s = ""
        for attribute, value in d.items():
            s += '      {} : {}\n'.format(attribute, value)
        return s.rstrip()

    def __repr__(self):
        string = "Strategy\n"
        string += "  - Active constraints:\n"
        string += self.__sprint_dict(self.active_constraints)
        if len(self.int_vars) > 0:
            string += "  - Integer variables values:\n"
            string += self.__sprint_dict(self.int_vars)
        return string

    def __hash__(self):
        """Overrides default hash implementation"""
        f_int_vars = frozenset(self.int_vars)
        f_active_constraints = frozenset(self.active_constraints)
        return hash((f_active_constraints, f_int_vars))

    def __eq__(self, other):
        """Overrides the default equality implementation"""
        if isinstance(other, Strategy):

            # Compare active constraints
            same_active_constraints = \
                self._compare_arrays_dict(self.active_constraints,
                                          other.active_constraints)

            # Compare integer variables
            same_int_vars = self._compare_arrays_dict(self.int_vars,
                                                      other.int_vars)

            return same_active_constraints and same_int_vars
        else:
            return False


def unique_strategies(strategies):
    """
    Extract unique strategies from array of strategies.

    Parameters
    ----------
    strategies : Strategy list
        Strategies to be processed.

    Returns
    -------
    Strategy array :
        Unique strategies.
    """
    return list(set(strategies))


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
