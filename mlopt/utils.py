import numpy as np
import scipy.io as spio
import cvxpy as cp
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
    for c in df.columns.values:
        if isinstance(df[c][0], list):
            # If list add length
            n += len(df[c][0])
        else:
            # If number add 1
            n += 1

    return n


# TODO: Fix creation from matlab file
#  def read_mat(filepath):
#      # Load file
#      m = spio.loadmat(filepath)
#
#      # Convert matrices
#      c = m['c'].T.flatten().astype(float)
#      Aeq = m['Aeq'].astype(float).tocsc()
#      beq = m['beq'].T.flatten().astype(float)
#      Aineq = m['Aineq'].astype(float).tocsc()
#      bineq = m['bineq'].T.flatten().astype(float)
#      if len(m['int_idx']) > 0:
#          int_idx = m['int_idx'].T.flatten().astype(int)
#      else:
#          int_idx = np.array([], dtype=int)
#
#      # Create cvxpy problem
#      x = cp.Variable(len(c))
#      cost = c * x
#      constraints = []
#
#
#
#      return problem_data(c, l, A, u, int_idx)
