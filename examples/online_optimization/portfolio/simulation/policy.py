from abc import ABCMeta, abstractmethod
import pandas as pd
import scipy.sparse as spa
import numpy as np
import datetime as dt
import cvxpy as cp


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
                 lambda_cost=None,
                 borrow_cost=0.0001):
        self.returns = returns
        self.risk_model = risk_model
        if lambda_cost is None:
            self.lambda_cost = {'risk': 50,
                                'borrow': 0.0001,
                                'norm1_trade': 0.01,
                                'norm0_trade': 1.}
        else:
            self.lambda_cost = lambda_cost
        lam = self.lambda_cost  # More readable code

        self.borrow_cost = borrow_cost

        # Store times
        self.times = returns.index
        self.periods = periods

        # Initialize problem
        n = len(returns.columns)
        k = len(risk_model['exposures'].columns)

        # Parameters
        hat_r = [cp.Parameter(n) for t in range(self.periods)]
        w_init = cp.Parameter(n)
        F = cp.Parameter((n, k))
        Sigma_F = cp.Parameter((k, k), PSD=True)
        sqrt_D = cp.Parameter(n)

        # Formulate problem
        w = [cp.Variable(n) for t in range(self.periods + 1)]

        # Define cost components
        cost = 0
        constraints = [w[0] == w_init]
        for t in range(1, self.periods + 1):

            risk_cost = lam['risk'] * (
                cp.quad_form(F.T * w[t], Sigma_F) +
                cp.sum_squares(cp.multiply(sqrt_D, w[t])))

            holding_cost = lam['borrow'] * \
                cp.sum(self.borrow_cost * cp.neg(w[t]))

            transaction_cost = lam['norm1_trade'] * cp.norm(w[t] - w[t-1], 1)

            cost += hat_r[t-1] * w[t] + \
                - risk_cost - holding_cost - transaction_cost

            constraints += [cp.sum(w[t]) == 1.]

        self.problem = cp.Problem(cp.Maximize(cost), constraints)

        # Store values
        self.vars = {'w': w}
        self.params = {}
        self.params['risk'] = {'F': F,
                               'Sigma_F': Sigma_F,
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
        Sigma_F = \
            np.diag(self.risk_model['sigma_factors'].loc[month].values)
        sqrt_D = np.sqrt(self.risk_model['idyos'].loc[month].values)

        # Evaluate parameters
        self.params['risk']['F'].value = F
        self.params['risk']['Sigma_F'].value = Sigma_F
        self.params['risk']['sqrt_D'].value = sqrt_D
        self.params['w_init'].value = w_init

    def trades(self, portfolio, t=pd.datetime.today()):
        """
        Solve Markowiz portfolio problem with cvxpy
        """

        # Evaluate parameters
        self.evaluate_params(portfolio, t)

        # Solve
        self.problem.solve(solver=cp.GUROBI)

        if self.problem.status not in cp.settings.SOLUTION_PRESENT:
            print("Problem in computing the solution")
            import ipdb; ipdb.set_trace()

        # Transform allocaitons into transactions
        return pd.Series(sum(portfolio) * (self.vars['w'][1].value -
                                           self.params['w_init'].value),
                         index=portfolio.index)
