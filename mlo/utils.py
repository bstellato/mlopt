import numpy as np
import scipy.sparse as spa
from cvxopt.base import matrix, spmatrix
from cvxopt.modeling import op
from mlo.problem import ProblemData


def read_mps(filepath):
    # Read problem from file path
    mat_form = op().fromfile(filepath)._inmatrixform(format='sparse')
    assert mat_form
    lp, _, _ = mat_form

    # Get variables
    x = lp.variables()[0]

    # Cost
    c = lp.objective._linear._coeff[x]

    # Inequalities
    inequalities = lp._inequalities
    G = inequalities[0]._f._linear._coeff[x]
    h = -inequalities[0]._f._constant

    # Equalities
    equalities = lp._equalities
    A, b = None, None
    if equalities:
        A = equalities[0]._f._linear._coeff[x]
        b = -equalities[0]._f._constant
    else:
        A = spmatrix(0.0, [], [],  (0, len(x)))  # CRITICAL
        b = matrix(0.0, (0, 1))

    c = np.array(c).flatten()
    G = np.array(G)
    h = np.array(h).flatten()
    A = np.array(A)
    b = np.array(b).flatten()

    int_idx =

    return ProblemData(c, l, A, u, int_idx)
