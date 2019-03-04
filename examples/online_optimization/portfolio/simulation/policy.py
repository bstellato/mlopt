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


class SinglePeriod(BasePolicy):
    """
    Single period optimization
    """

    def __init__(self, returns, risk_model,
                 lambda_cost=None,
                 borrow_cost=0.0001):
        self.returns = returns
        self.risk_model = risk_model
        if lambda_cost is None:
            self.lambda_cost = {'risk': 40.,
                                'borrow': 0.0001,
                                'norm1_trade': 0.01,
                                'norm0_trade': 0.01}
        else:
            self.lambda_cost = lambda_cost

        self.borrow_cost = borrow_cost

        # Initialize problem
        n = len(returns.columns)
        k = len(risk_model['exposures'].columns)

        # Parameters
        hat_r = cp.Parameter(n)
        w_init = cp.Parameter(n)
        F = cp.Parameter((n, k))
        Sigma_F = cp.Parameter((k, k), PSD=True)
        sqrt_D = cp.Parameter(n)

        # Formulate problem
        w = cp.Variable(n)

        lam = self.lambda_cost

        # Define cost components
        risk_cost = \
            cp.quad_form(F.T * w, Sigma_F) + \
            cp.sum_squares(cp.multiply(sqrt_D, w))
        risk_cost *= lam['risk']
        holding_cost = lam['borrow'] * \
            cp.sum(self.borrow_cost * cp.neg(w))
        transaction_cost = lam['norm1_trade'] * cp.norm(w - w_init, 1)

        # Define full cost and constraints
        cost = hat_r * w - risk_cost - holding_cost - transaction_cost
        constraints = [cp.sum(w) == 1]
        self.problem = cp.Problem(cp.Maximize(cost), constraints)

        # Store values
        self.params, self.vars = {}, {}
        self.params['hat_r'] = hat_r
        self.params['w_init'] = w_init
        self.params['F'] = F
        self.params['Sigma_F'] = Sigma_F
        self.params['sqrt_D'] = sqrt_D
        self.vars['w'] = w

    def evaluate_params(self, portfolio, t):
        """Evaluate parameters for returns and risk model"""
        # Get initial value
        value = sum(portfolio)
        w_init = (portfolio / value).values

        # r estimate
        hat_r = self.returns.loc[t].values

        # Risk estimate
        month = dt.date(t.year, t.month, 1)  # Get first day of month
        F = self.risk_model['exposures'].loc[month].values
        Sigma_F = \
            np.diag(self.risk_model['sigma_factors'].loc[month].values)
        sqrt_D = np.sqrt(self.risk_model['idyos'].loc[month].values)

        # Evaluate parameters
        self.params['F'].value = F
        self.params['Sigma_F'].value = Sigma_F
        self.params['sqrt_D'].value = sqrt_D
        self.params['hat_r'].value = hat_r
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
        return pd.Series(sum(portfolio) * (self.vars['w'].value -
                                           self.params['w_init'].value),
                         index=portfolio.index)
