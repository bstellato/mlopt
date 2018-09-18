import numpy as np
import pandas as pd


def accuracy(strategy_pred,
             strategy_test):
    """
    Accuracy comparison between predicted and test strategies

    Args:
        strategy_pred (Strategy list): List of predicted strategies.
        strategy_pred (Strategy list): List of test strategies.

    Returns:
        float: Fraction of correct over total strategies compared.
        numpy array: Boolean vector indicating which strategy is correct.

    """
    assert len(strategy_pred) == len(strategy_test)
    n_total = len(strategy_pred)
    idx_correct = np.zeros(n_total)
    for i in range(n_total):
        if strategy_pred[i] == strategy_test[i]:
            idx_correct[i] = 1

    return np.sum(idx_correct) / n_total, idx_correct


def eval_performance(theta, lnr, problem, enc2strategy, k=1):
    """
    Evaluate predictor performance

    Parameters
    ----------
    theta : DataFrame
        Data to predict.
    lnr : Learner
        Learner.
    problem : OptimizationProblem
        Optimization problem.
    enc2strategy : Strategy list
        Mapping between encoding to the strategy.
    k : int, optional
        Best k predicted values to pick. Defaults to 0.
    """

    print("Performance evaluation")
    print("Compute active constraints over test set")

    # Get strategy for each point
    x_test, time_test, strategy_test = problem.solve(theta)

    # Get predicted strategy for each point
    x_pred, time_pred, strategy_pred = predict_best(theta, k, lnr,
                                                    problem, enc2strategy)

    # TODO: Fix rest














