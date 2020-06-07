import unittest
import numpy as np
import cvxpy as cp
import pandas as pd
from mlopt.optimizer import Optimizer
from mlopt.settings import PYTORCH
from mlopt.sampling import uniform_sphere_sample


def sample_portfolio(n, k, T=5, N=100):
    """Sample portfolio parameters."""

    rad = 0.01

    # mean values
    np.random.seed(0)
    SigmaF_FT_bar = np.random.randn(k, n)
    sqrt_D_bar = np.random.rand(n)
    #  Sigma_F_diag_bar = np.random.rand(k)
    hat_r_bar = np.random.rand(n)
    w_init_bar = np.random.rand(n)
    w_init_bar /= np.sum(w_init_bar)

    df = pd.DataFrame()
    for i in range(N):

        w_init = uniform_sphere_sample(w_init_bar, rad).flatten()
        w_init /= np.sum(w_init)

        SigmaF_FT = uniform_sphere_sample(SigmaF_FT_bar.flatten(),
                                          rad).reshape(k, n)

        x = pd.Series(
            {
                #  "F": F,
                "sqrt_D": uniform_sphere_sample(sqrt_D_bar, rad).flatten(),
                "SigmaF_FT": SigmaF_FT,
                "w_init": w_init,
            }
        )

        for t in range(1, T + 1):
            x["hat_r_%s" % str(t)] = uniform_sphere_sample(hat_r_bar,
                                                           rad).flatten()

        df = df.append(x, ignore_index=True)

    return df


class TestMatrixParams(unittest.TestCase):
    def test_matrix_multiperiod_portfolio(self):
        np.random.seed(1)

        k = 10
        n = 50
        T = 5
        borrow_cost = 0.0001
        lam = {
            "risk": 50,
            "borrow": 0.0001,
            "norm1_trade": 0.01,
            "norm0_trade": 1.0,
        }

        # Parameters
        hat_r = [
            cp.Parameter(n, name="hat_r_%s" % str(t)) for t in range(1, T + 1)
        ]
        w_init = cp.Parameter(n, name="w_init")
        #  F = cp.Parameter((n, k), name="F")
        sqrt_SigmaF_FT = cp.Parameter((k, n), name="SigmaF_FT")
        sqrt_D = cp.Parameter(n, name="sqrt_D")

        # Formulate problem
        w = [cp.Variable(n) for t in range(T + 1)]

        # Define cost components
        cost = 0
        constraints = [w[0] == w_init]
        for t in range(1, T + 1):

            risk_cost = lam["risk"] * (
                #  cp.quad_form(F.T * w[t], Sigma_F)
                cp.sum_squares(sqrt_SigmaF_FT @ w[t]) +
                cp.sum_squares(cp.multiply(sqrt_D, w[t]))
            )

            holding_cost = lam["borrow"] * cp.sum(borrow_cost * cp.neg(w[t]))

            transaction_cost = lam["norm1_trade"] * cp.norm(w[t] - w[t - 1], 1)

            cost += (
                hat_r[t - 1] @ w[t]
                - risk_cost
                - holding_cost
                - transaction_cost
            )

            constraints += [cp.sum(w[t]) == 1.0]

        # Define optimizer
        problem = cp.Problem(cp.Maximize(cost), constraints)
        m = Optimizer(problem)

        self.assertTrue(m._problem.parameters_in_matrices)

        # Sample parameters
        df_train = sample_portfolio(n, k, N=1000)

        # Train and test using pytorch
        m.train(df_train, filter_strategies=True, parallel=True,
                learner=PYTORCH)

        # Assert fewer strategies than training samples
        self.assertTrue(len(m.encoding) < len(df_train))
        self.assertTrue(len(m.encoding) > 1)
