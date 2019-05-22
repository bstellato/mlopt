from abc import ABCMeta, abstractmethod
import pandas as pd
import scipy.sparse as spa
import numpy as np
import datetime as dt
import cvxpy as cp
import online_optimization.portfolio.simulation.settings as stg


class BasePolicy(object):
    """ Base class for a trading policy. """
    __metaclass__ = ABCMeta

    def __init__(self):
        self.costs = []
        self.constraints = []

    @abstractmethod
    def trades(self, portfolio, t=pd.datetime.today()):
        """Trades list given current portfolio and time t.
        """
        return NotImplemented

    def _nulltrade(self, portfolio):
        return pd.Series(index=portfolio.index, data=0.)

    #  def get_rounded_trades(self, portfolio, prices, t):
    #      """Get trades vector as number of shares, rounded to integers."""
    #      return np.round(self.get_trades(portfolio, t) /
    #                      prices.loc[t])[:-1]


class Hold(BasePolicy):
    """Hold initial portfolio.
    """

    def trades(self, portfolio, t=pd.datetime.today()):
        return self._nulltrade(portfolio)


class Optimal(BasePolicy):
    """
    Multi-period optimization
    """

    def __init__(self, returns, risk_model,
                 periods=1,  # Default one period
                 k=None,  # Sparsity
                 lambda_cost=None):
        self.returns = returns
        self.risk_model = risk_model
        if lambda_cost is None:
            self.lambda_cost = {'risk': stg.RISK_COST,
                                'borrow': stg.BORROW_WEIGHT_COST,
                                #  'norm0_trade': stg.NORM0_TRADE_COST,
                                'norm1_trade': stg.NORM1_TRADE_COST,
                                #  'norm2_trade': stg.NORM2_TRADE_COST
                                }
        else:
            self.lambda_cost = lambda_cost
        lam = self.lambda_cost  # More readable code

        self.borrow_cost = stg.BORROW_COST

        # Store times
        self.times = returns.index
        self.periods = periods

        # Initialize problem
        n = len(returns.columns)
        m = len(risk_model['exposures'].columns)

        # Parameters
        hat_r = [cp.Parameter(n) for t in range(self.periods)]
        w_init = cp.Parameter(n)
        F = cp.Parameter((n, m))
        sqrt_Sigma_F = cp.Parameter(m)
        sqrt_D = cp.Parameter(n)

        # Formulate problem
        w = [cp.Variable(n) for t in range(self.periods + 1)]

        if k is not None:
            # Sparsity constraints
            s = [cp.Variable(n, boolean=True) for t in range(self.periods)]

        # Define cost components
        cost = 0
        constraints = [w[0] == w_init]
        for t in range(1, self.periods + 1):

            risk_cost = lam['risk'] * (
                cp.sum_squares(cp.multiply(sqrt_Sigma_F, F.T * w[t])) +
                cp.sum_squares(cp.multiply(sqrt_D, w[t])))

            holding_cost = lam['borrow'] * \
                cp.sum(stg.BORROW_COST * cp.neg(w[t]))

            transaction_cost = \
                lam['norm1_trade'] * cp.norm(w[t] - w[t-1], 1)
            #  lam['norm2_trade'] * cp.sum_squares(w[t] - w[t-1])  # + \

            cost += hat_r[t-1] * w[t] + \
                - risk_cost - holding_cost - transaction_cost

            constraints += [cp.sum(w[t]) == 1.]

            if k is not None:
                # Cardinality constraint (big-M)
                constraints += [-s[t-1] <= w[t] - w[t-1], w[t] - w[t-1] <= s[t-1],
                                cp.sum(s[t-1]) <= k]

        self.problem = cp.Problem(cp.Maximize(cost), constraints)

        # Store values
        self.vars = {'w': w}
        self.params = {}
        self.params['risk'] = {'F': F,
                               'sqrt_Sigma_F': sqrt_Sigma_F,
                               'sqrt_D': sqrt_D}
        self.params['w_init'] = w_init
        self.params['hat_r'] = hat_r

    def evaluate_params(self, portfolio, t):
        """Evaluate parameters for returns and risk model"""
        # Get initial value
        value = sum(portfolio)
        w_init = (portfolio / value).values

        # Get times in window of lookahead periods
        times = self.times[self.times > t][:self.periods]
        assert len(times) == self.periods

        # r estimates
        for i in range(self.periods):
            self.params['hat_r'][i].value = self.returns.loc[times[i]].values

        # Risk estimate
        month = dt.date(t.year, t.month, 1)  # Get first day of month
        F = self.risk_model['exposures'].loc[month].values
        sqrt_Sigma_F = \
            np.sqrt(self.risk_model['sigma_factors'].loc[month].values)
        sqrt_D = np.sqrt(self.risk_model['idyos'].loc[month].values)

        # Evaluate parameters
        self.params['risk']['F'].value = F
        self.params['risk']['sqrt_Sigma_F'].value = sqrt_Sigma_F
        self.params['risk']['sqrt_D'].value = sqrt_D
        self.params['w_init'].value = w_init

    def trades(self, portfolio, t=pd.datetime.today(), verbose=False):
        """
        Solve Markowiz portfolio problem with cvxpy
        """

        # Evaluate parameters
        self.evaluate_params(portfolio, t)

        # Solve
        self.problem.solve(solver=cp.GUROBI, verbose=verbose)

        if self.problem.status not in cp.settings.SOLUTION_PRESENT:
            print("Problem in computing the solution")
            import ipdb; ipdb.set_trace()

        # Transform allocaitons into transactions
        return pd.Series(sum(portfolio) * (self.vars['w'][1].value -
                                           self.params['w_init'].value),
                         index=portfolio.index)
