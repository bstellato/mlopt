import numpy as np
import pandas as pd
from tqdm import tqdm
from .settings import TOL


def accuracy(strategy_pred, strategy_test):
    """
    Accuracy comparison between predicted and test strategies

    Parameters
    ----------
    strategy_pred : Strategy list
        List of predicted strategies.
    strategy_pred : Strategy list
        List of test strategies.

    Returns
    -------
    float:
        Fraction of correct over total strategies compared.
    numpy array:
        Boolean vector indicating which strategy is correct.
    """
    assert len(strategy_pred) == len(strategy_test)
    n_total = len(strategy_pred)
    idx_correct = np.zeros(n_total, dtype=int)
    for i in range(n_total):
        if strategy_pred[i] == strategy_test[i]:
            idx_correct[i] = 1

    return np.sum(idx_correct) / n_total, idx_correct


def eval_performance(theta, learner, problem, enc2strategy, k=1):
    """
    Evaluate predictor performance

    Parameters
    ----------
    theta : DataFrame
        Data to predict.
    learner : Learner
        Learner.
    problem : OptimizationProblem
        Optimization problem.
    enc2strategy : Strategy list
        Mapping between encoding to the strategy.
    k : int, optional
        Best k predicted values to pick. Defaults to 0.
    """

    print("Performance evaluation")
    # Get strategy for each point
    x_test, time_test, strategy_test = \
        problem.solve_parametric(theta, message="Compute active " +
                                                "constraints for test set")

    # Get predicted strategy for each point
    x_pred, time_pred, strategy_pred = learner.predict_best_points(
        theta, problem, k, enc2strategy,
        message="Predict active constraints for test set"
    )

    num_test = len(theta)
    num_train = learner.n_train  # Number of training samples from learner
    n_theta = theta.shape[1]  # Parameters dimension
    n_active_sets = len(enc2strategy)  # Number of active sets

    # Compute infeasibility and optimality for each problem
    # NB. We need to populate the problem for each point
    infeas = []
    subopt = []
    time_comp = []
    for i in range(num_test):
        problem.populate(theta.iloc[i, :])
        infeas.append(problem.infeasibility(x_pred[i]))
        subopt.append(problem.suboptimality(x_pred[i], x_test[i]))
        time_comp.append(1 - time_pred[i] / time_test[i])

    # Convert to array
    infeas = np.array(infeas)
    subopt = np.array(subopt)
    time_comp = np.array(time_comp)

    # Get number of variables and constraints from one of the problems
    num_var = len(problem.data['c'])
    num_constr = len(problem.data['l'])

    # accuracy
    test_accuracy, idx_correct = accuracy(strategy_pred, strategy_test)

    # Create dataframes to return
    df = pd.DataFrame(
        {
            "problem": [problem.name],
            "k": [k],
            "num_var": [num_var],
            "num_constr": [num_constr],
            "num_test": [num_test],
            "num_train": [num_train],
            "n_theta": [n_theta],
            "n_corect": [np.sum(idx_correct)],
            "n_active_sets": [n_active_sets],
            "accuracy": [test_accuracy],
            "n_infeas": [np.sum(infeas >= TOL)],
            "avg_infeas": [np.mean(infeas)],
            "avg_subopt": [np.mean(subopt[np.where(infeas <= TOL)[0]])],
            "max_infeas": [np.max(infeas)],
            "max_subopt": [np.max(subopt)],
            "avg_time_improv": [np.mean(time_comp)],
            "max_time_improv": [np.max(time_comp)],
        }
    )
    # Add radius info if problem has it.
    # TODO: We should remove it later
    try:
        df["radius"] = [problem.radius]
    except AttributeError:
        pass

    df_detail = pd.DataFrame(
        {
            "problem": [problem.name] * num_test,
            "correct": idx_correct,
            "infeas": infeas,
            "subopt": subopt,
            "time_improvement": time_comp,
        }
    )

    return df, df_detail, x_pred, x_test


def store(results, file_name):
    """
    Store results as csv files

    Parameters
    ----------
    results : array_like
        Results to be stored.
    file_name : string
        File name.
    """
    for i in tqdm(range(len(results)), desc="Exporting results"):
        results[i].to_csv(file_name + "%d.csv" % i)
