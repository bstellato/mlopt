import numpy as np
import scipy.io as spio
import cvxpy as cvx
import cvxpy.settings as s
import scipy.sparse as spa


def num_dataframe_features(df):
    """
    Get number of features in dataframe
    where cells contain tuples.

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe.

    Returns
    -------
    int
        Number of features.
    """
        n = 0
        for c in X.columns.values:
            if isinstance(X[c][0], list):
                # If list add length
                n += len(X[c][0])
            else:
                # If number add 1
                n += 1

    return n

def problem_data(c, l, A, u, int_idx=None):
    """Create problem data dictionary"""
    data = {'c': c,
            'l': l,
            'A': A,
            'u': u}
    if int_idx is not None:
        data['int_idx'] = int_idx
    else:
        data['int_idx'] = np.array([], dtype=int)
    return data


def read_mat(filepath):
    # Load file
    m = spio.loadmat(filepath)

    # Convert matrices
    c = m['c'].T.flatten().astype(float)
    A = m['A'].astype(float).tocsc()
    l = m['l'].T.flatten().astype(float)
    u = m['u'].T.flatten().astype(float)
    if len(m['int_idx']) > 0:
        int_idx = m['int_idx'].T.flatten().astype(int)
    else:
        int_idx = np.array([], dtype=int)

    return problem_data(c, l, A, u, int_idx)


def cvxpy2data(problem):

    #  data = problem.get_problem_data(cvx.OSQP)[0]  # Get problem data
    data = problem.get_problem_data(cvx.CPLEX)[0]  # Get problem data
    #  data = problem.get_problem_data(cvx.GLPK_MI)[0]  # Get problem data

    int_idx = data['int_vars_idx']
    c = data[s.Q]
    A = spa.vstack([data[s.A], data[s.F]]).tocsc()
    u = np.concatenate((data[s.B], data[s.G]))
    l = np.concatenate([data[s.B], -np.inf*np.ones(data[s.G].shape)])

    return problem_data(c, l, A, u, int_idx)

