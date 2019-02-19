import numpy as np
from multiprocessing import cpu_count
#  from pathos.multiprocessing import cpu_count
from mlopt.settings import INFEAS_TOL, SUBOPT_TOL, TIGHT_CONSTRAINTS_TOL
import os
import pandas as pd
import mlopt
import logging


def args_norms(expr):
    """Calculate norm of the arguments in a cvxpy expression"""
    if expr.args:
        norms = []
        # Expression contains arguments
        for arg in expr.args:
            norms += args_norms(arg)
        return norms
    else:
        return [np.linalg.norm(expr.value)]


def tight_components(con):
    """Return which components are tight in the constraints."""
    return np.abs(con.expr.value) <= TIGHT_CONSTRAINTS_TOL


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
        n_cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    except KeyError:
        n_cpus = cpu_count()
    n_proc = min(max_n, n_cpus)

    return n_proc


def add_details(df, **kwargs):
    """
    Add details to dataframe
    """
    len_df = len(df)  # Dataframe length

    for key, val in kwargs.items():
        df[key] = [val] * len_df


def benchmark(m,  # Optimizer
              data_file,
              theta_bar,
              sample_fn,
              dims,
              trees=True
              ):
    """
    Perform benchmark

    Parameters
    ----------
    m : Optimizer
        Problem optimizer.
    data_file : string
        Name of the data file.
    theta_bar : array or dict
        Average value of optimizer.
    sample_fn : Function
        Sampling function.
    dims : dict
        Problem dimensions.
    trees : bool, optional
        Whether or not to train the trees. Defaults to true.
    """

    # Reset random seed
    np.random.seed(1)

    # Get test elements
    theta_test = sample_fn(100)

    data_file_general = data_file + "_general.csv"
    data_file_detail = data_file + "_detail.csv"

    # Loading data points
    already_run = os.path.isfile(data_file_general) and \
        os.path.isfile(data_file_detail)
    if already_run:
        logging.info("Loading data %s" % data_file)
        general = pd.read_csv(data_file_general)
        detail = pd.read_csv(data_file_detail)
    else:
        logging.info("Perform training for %s" % data_file)

        logging.info("Training NN")
        logging.info("-----------\n")

        # Train neural network
        m.train(sampling_fn=sample_fn,
                parallel=True,
                learner=mlopt.PYTORCH)

        general, detail = m.performance(theta_test, parallel=True)

        # Fix dataframe by adding elements
        add_details(general, predictor="NN", **dims)
        add_details(detail, predictor="NN", **dims)

        #  Train and test using optimal trees
        if trees:

            logging.info("Training OCT")
            logging.info("-----------\n")

            # OCT
            m.train(
                    parallel=True,
                    learner=mlopt.OPTIMAL_TREE,
                    hyperplanes=False,
                    save_svg=True)
            oct_general, oct_detail = m.performance(theta_test, parallel=True)
            add_details(oct_general, predictor="OCT", **dims)
            add_details(oct_detail, predictor="OCT", **dims)

            #  Combine and store partial results
            general = general.append(oct_general)
            detail = detail.append(oct_detail)

            logging.info("Training OCT-H")
            logging.info("-----------\n")

            # OCT-H
            m.train(
                    parallel=True,
                    learner=mlopt.OPTIMAL_TREE,
                    hyperplanes=True,
                    save_svg=True)
            octh_general, octh_detail = m.performance(theta_test,
                                                      parallel=True)
            add_details(octh_general, predictor="OCT-H", **dims)
            add_details(octh_detail, predictor="OCT-H", **dims)

            #  Combine and store partial results
            general = general.append(octh_general)
            detail = detail.append(octh_detail)

        # Store to csv
        general.to_csv(data_file_general, index=False)
        detail.to_csv(data_file_detail, index=False)

    return general, detail


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
    n = 0
    for c in df.columns.values:
        if isinstance(df[c].iloc[0], list):  # If list add length
            n += len(df[c].iloc[0])
        else:  # If number add 1
            n += 1
    return n


def pandas2array(X):
    """
    Unroll dataframe elements to construct 2d array in case of
    cells containing tuples.
    """

    if isinstance(X, np.ndarray):
        # Already numpy array. Return it.
        return X
    else:
        # get number of datapoints
        n_data = len(X)
        # Get dimensions by inspecting first row
        n = n_features(X)

        # Allocate full vector
        X_new = np.empty((0, n))

        # Unroll
        # TODO: Speedup this process
        for i in range(n_data):
            x_temp = np.array([])
            x_data = X.iloc[i, :].values
            for i in x_data:
                if isinstance(i, list):
                    x_temp = np.concatenate((x_temp, np.array(i)))
                else:
                    x_temp = np.append(x_temp, i)
            X_new = np.vstack((X_new, x_temp))

    return X_new


def suboptimality(cost_pred, cost_test):
    """Compute suboptimality"""
    return (cost_pred - cost_test)/(np.abs(cost_test) + 1e-10)


def accuracy(results_pred, results_test):
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
            if r_pred['infeasibility'] <= INFEAS_TOL:
                # Check cost function value
                subopt = suboptimality(r_pred['cost'], r_test['cost'])
                if subopt <= SUBOPT_TOL:
                    idx_correct[i] = 1

    return np.sum(idx_correct) / n_points, idx_correct
