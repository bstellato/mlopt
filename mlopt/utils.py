import numpy as np
import cvxpy as cp
import os
import pandas as pd
import mlopt.settings as stg
import joblib


#  def args_norms(expr):
#      """Calculate norm of the arguments in a cvxpy expression"""
#      if expr.args:
#          norms = []
#          # Expression contains arguments
#          for arg in expr.args:
#              #  norms += args_norms(arg)
#              norms += [cp.norm(arg, np.inf).value]
#      else:
#          norms = [0.]
#      return norms


#  def tight_components(con):
#      """Return which components are tight in the constraints."""
#      #  rel_norm = np.amax([np.linalg.norm(np.atleast_1d(a.value), np.inf)
#      #                      for a in con.expr.args])
#      # If Equality Constraint => all tight
#      if type(con) in [Equality, Zero]:
#          return np.full(con.shape, True)
#
#      # Otherwise return violation
#      rel_norm = 1.0
#      return np.abs(con.expr.value) <= stg.TIGHT_CONSTRAINTS_TOL * (1 + rel_norm)


def get_n_processes(max_n=np.inf):
    """Get number of processes from current cps number

    Parameters
    ----------
    max_n: int
        Maximum number of processes.

    Returns
    -------
    float
        Number of processes to use.
    """

    try:
        # Check number of cpus if we are on a SLURM server
        n_cpus = int(os.environ["SLURM_NPROCS"])
    except KeyError:
        n_cpus = joblib.cpu_count()

    n_proc = min(max_n, n_cpus)

    return n_proc


def n_features(df):
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
    return sum(np.atleast_1d(x).size for x in df.iloc[0])

    #  n = 0
    #  for c in df.columns.values:
    #
    #      if isinstance(df[c].iloc[0], list):  # If list add length
    #          n += len(df[c].iloc[0])
    #      else:  # If number add 1
    #          n += 1
    #  return n


def pandas2array(X):
    """
    Unroll dataframe elements to construct 2d array in case of
    cells containing tuples.
    """

    if isinstance(X, np.ndarray):
        # Already numpy array. Return it.
        return X
    else:
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X).transpose()
        # get number of datapoints
        n_data = len(X)

        x_temp_list = []
        for i in range(n_data):
            x_temp_list.append(
                np.concatenate([np.atleast_1d(v).flatten()
                                for v in X.iloc[i].values])
                )
        X_new = np.vstack(x_temp_list)

    return X_new


def suboptimality(cost_pred, cost_test, sense):
    """Compute suboptimality"""
    if np.abs(cost_test) < stg.DIVISION_TOL:
        cost_norm = 1.
    else:
        cost_norm = np.abs(cost_test)

    if sense == cp.Minimize:
        return (cost_pred - cost_test)/cost_norm
    else:  # Maximize
        return (cost_test - cost_pred)/cost_norm


def accuracy(results_pred, results_test, sense):
    """
    Accuracy comparison between predicted and test results.

    Parameters
    ----------
    results_red : dictionary of predict results.
        List of predicted results.
    results_test : dictionary of test results.
        List of test results.

    Returns
    -------
    float:
        Fraction of correct over total strategies compared.
    numpy array:
        Boolean vector indicating which strategy is correct.
    numpy array:
        Boolean vector indicating which strategy is exact.
    """

    # Assert correctness by compariing solution cost and infeasibility
    n_points = len(results_pred)
    assert n_points == len(results_test)

    idx_correct = np.zeros(n_points, dtype=int)
    for i in range(n_points):
        r_pred = results_pred[i]
        r_test = results_test[i]
        # Check if prediction is correct
        if r_pred['strategy'] == r_test['strategy']:
            idx_correct[i] = 1
        else:
            # Check feasibility
            if r_pred['infeasibility'] <= stg.INFEAS_TOL:
                # Check cost function value
                subopt = suboptimality(r_pred['cost'], r_test['cost'], sense)
                if np.abs(subopt) <= stg.SUBOPT_TOL:
                    idx_correct[i] = 1

    return np.sum(idx_correct) / n_points, idx_correct
