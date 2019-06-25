import online_optimization.portfolio.simulation.settings as stg
import logging
import cvxpy as cp
import mlopt


def create_mlopt_problem(n, m, n_periods, k=None,
                         lambda_cost=None,
                         tight_constraints=True):

    if lambda_cost is None:
        lambda_cost = {'risk': stg.RISK_COST,
                       'borrow': stg.BORROW_COST,
                       #  'norm0_trade': stg.NORM0_TRADE_COST,
                       'norm1_trade': stg.NORM1_TRADE_COST}
    else:
        lambda_cost = lambda_cost

    # Parameters
    hat_r = [cp.Parameter(n, name="hat_r_%s" % (t + 1))
             for t in range(n_periods)]
    w_init = cp.Parameter(n, name="w_init")
    F = cp.Parameter((n, m), name="F")
    sqrt_Sigma_F = cp.Parameter(m, name="sqrt_Sigma_F")
    sqrt_D = cp.Parameter(n, name="sqrt_D")

    #  Sigma = psd_wrap(F * (Sigma_F * F.T) + cp.diag(cp.power(sqrt_D, 2)))

    # Formulate problem
    w = [cp.Variable(n) for t in range(n_periods + 1)]

    # Sparsity constraints
    if k is not None:
        s = [cp.Variable(n, boolean=True) for t in range(n_periods)]

    # Define cost components
    cost = 0
    constraints = [w[0] == w_init]
    for t in range(1, n_periods + 1):

        risk_cost = lambda_cost['risk'] * (            
            cp.sum_squares(cp.multiply(sqrt_Sigma_F, F.T * w[t])) +
            cp.sum_squares(cp.multiply(sqrt_D, w[t])))
        
        holding_cost = lambda_cost['borrow'] * \
            cp.sum(stg.BORROW_COST * cp.neg(w[t]))

        transaction_cost = \
            lambda_cost['norm1_trade'] * cp.norm(w[t] - w[t-1], 1)

        cost += \
            hat_r[t-1] * w[t] \
            - risk_cost \
            - holding_cost \
            - transaction_cost

        constraints += [cp.sum(w[t]) == 1.]

        if k is not None:
            # Cardinality constraint (big-M)
            constraints += [-s[t-1] <= w[t] - w[t-1], w[t] - w[t-1] <= s[t-1],
                            cp.sum(s[t-1]) <= k]

    return mlopt.Optimizer(cp.Maximize(cost), constraints,
                           log_level=logging.INFO,
                           #  verbose=True,
                           tight_constraints=tight_constraints,
                           parallel=True,
                           )


def get_problem_dimensions(df):
    
    n, m = df.iloc[0]['F'].shape

    return n, m
