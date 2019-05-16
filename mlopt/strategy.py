import numpy as np
import logging


class Strategy(object):
    """
    Solving strategy.

    Parameters
    ----------
    tight_constraints : dict of numpy int arrays
        Set of tight constraints. The keys are the CVXPY constraint ids.
        The values are numpy bool arrays (True/False for tight/non tight).
    int_vars : dict of numpy int arrays
        Value of the integer variables. The keys are CVXPY variable id.
        The values are numpy int arrays.
    """

    def __init__(self, tight_constraints, int_vars):
        # Check tight constraints
        for _, v in tight_constraints.items():
            if np.any(np.logical_or(v < 0, v > 1)):
                err = "Tight constraints vector is not boolean."
                logging.error(err)
                raise ValueError(err)

        # Check that integer variables are non negative
        #  for _, v in int_vars.items():
        #      if np.any(v < 0):
        #          raise ValueError("Integer variables vector " +
        #                           "has negative entries.")

        # Check that tight constraints are not
        self.tight_constraints = tight_constraints
        self.int_vars = int_vars

    def _compare_arrays_dict(self, d1, d2):
        """Compare dictionaries of numpy arrays"""
        if len(d1) != len(d2):
            return False
        for key in d1.keys():
            try:
                isequal = np.array_equal(d1[key], d2[key])
            except KeyError:
                return False
            if not isequal:
                return False
        return True

    def __sprint_dict(self, d):
        s = ""
        for attribute, value in d.items():
            s += '      {:>5}: {}\n'.format(attribute, value)
        return s.rstrip()

    def __repr__(self):
        string = "Strategy\n"
        string += "  - Tight constraints:\n"
        string += "         id: elements\n"
        string += self.__sprint_dict(self.tight_constraints)
        string += "\n"
        if len(self.int_vars) > 0:
            string += "  - Integer variables values:\n"
            string += "         id: elements\n"
            string += self.__sprint_dict(self.int_vars)
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
                self._compare_arrays_dict(self.tight_constraints,
                                          other.tight_constraints)

            if not same_tight_constraints:
                return False

            # Compare integer variables
            same_int_vars = self._compare_arrays_dict(self.int_vars,
                                                      other.int_vars)

            return same_tight_constraints and same_int_vars
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
    Strategy set :
        Unique strategies.
    """
    # Using set
    #  unique = list(set(strategies))

    # Using list
    unique = []
    # traverse for all elements
    for x in strategies:
        # check if exists in unique_list or not
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
    logging.info("Encoding strategies")
    N = len(strategies)

    logging.info("Getting unique set of strategies")
    unique = unique_strategies(strategies)
    n_unique_strategies = len(unique)
    logging.info("Found %d unique strategies" % n_unique_strategies)

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
    if s.tight_constraints:
        s_tight = np.concatenate(
            [np.atleast_1d(v) for k, v in sorted(s.tight_constraints.items())]
        )
    else:
        s_tight = np.array([])
    if s.int_vars:
        s_int = np.concatenate(
            [np.atleast_1d(v) for k, v in sorted(s.int_vars.items())]
        )
    else:
        s_int = np.array([])

    return np.concatenate((s_tight, s_int))


def strategy_distance(a, b):
    """Compute manhattan distance between strategy a and b."""
    # Convert strategies to array
    a_array = strategy2array(a)
    b_array = strategy2array(b)

    return np.linalg.norm(a_array - b_array, 1) / len(a_array)
