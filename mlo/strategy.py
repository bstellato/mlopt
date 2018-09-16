# Define strategy
import numpy as np


class Strategy:
    """
    Solving strategy

    Parameters
    ----------
    int_vars: numpy array
        Value of the integer variables
    active_constraints: set
        Set of active constraints
    """

    def __init__(self, int_vars, active_constraints=set()):
        self.int_vars
        self.active_constraints

    def __eq__(self, other):
        """Overrides the default equality implementation"""
        if isinstance(other, Strategy):
            same_int_vars = np.array_equal(self.int_vars, other.int_vars)
            same_active_constraints = (
                self.active_constraints == other.active_constraints
            )
            return same_int_vars + same_active_constraints
        return False


def unique_strategies(strategies):
    n_int_var = len(strategies[0].int_vars)

    # Construct vector of vectors and get unique elements
    # NB. We need to convert the active constraints from set
    # to numpy arrays to make it work
    strategy_vecs = np.unique(
        np.array(
            [
                np.concatenate((s.int_vars, np.array(s.active_constraints)))
                for s in strategies
            ]
        ),
        axis=0,
    )

    # Get unique vectors
    return [Strategy(s[:n_int_var], s[n_int_var:]) for s in strategy_vecs]
