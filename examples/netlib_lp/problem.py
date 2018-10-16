import cvxpy as cvx
import pandas as pd
import mlopt
import importlib
importlib.reload(mlopt)


class Inventory(mlopt.Problem):
    def __init__(self, T, M, K, radius, bin_vars=False):
        self.name = "inventory"
        self.T = T  # Horizon
        self.M = M  # Maximum ordering capacity
        self.K = K  # Fixed ordering cost
        self.radius = radius  # Radius for sampling

        # Define model in cvxpy
        x = cvx.Variable(T+1)
        u = cvx.Variable(T)
        y = cvx.Variable(T+1)  # Auxiliary y = max(h * x, - p * x)
        self.vars = {'x': x,
                     'u': u,
                     'y': y}
        if bin_vars:
            v = cvx.Variable(T, integer=True)
            self.vars['v'] = v

        # Define parameters
        x0 = cvx.Parameter(nonneg=True)
        h = cvx.Parameter(nonneg=True)
        p = cvx.Parameter(nonneg=True)
        c = cvx.Parameter(nonneg=True)
        d = cvx.Parameter(T, nonneg=True)
        self.params = {'x0': x0, 'h': h, 'p': p,
                       'c': c, 'd': d}

        # Constaints
        constraints = []
        constraints += [x[0] == x0]
        constraints += [y >= h * x, y >= -p * x]
        for t in range(T):
            constraints += [x[t+1] == x[t] + u[t] - d[t]]
        constraints += [u >= 0]
        if bin_vars:
            constraints += [u <= M * v]
            constraints += [0 <= v, v <= 1]  # Binary variables
        else:
            constraints += [u <= M]

        # Objective
        cost = cvx.sum(y) + c * cvx.sum(u)
        if bin_vars:
            cost += K * cvx.sum(v)

        # Define problem
        self.cvxpy_problem = cvx.Problem(cvx.Minimize(cost), constraints)

    def populate(self, theta):
        """
        Populate problem using parameter theta.
        """

        # Get parameters from dataframe
        self.params['h'].value = theta["h"]
        self.params['p'].value = theta["p"]
        self.params['c'].value = theta["c"]
        self.params['x0'].value = theta["x0"]
        self.params['d'].value = theta.iloc[4:].values

        # Get new problem data
        self.data = mlopt.cvxpy2data(self.cvxpy_problem)

    def sample(self, theta_bar, N=100):

        # Sample points from multivariate ball
        X = mlopt.uniform_sphere_sample(theta_bar, self.radius, N=N)

        df = pd.DataFrame({'h': X[:, 0],
                           'p': X[:, 1],
                           'c': X[:, 2],
                           'x0': X[:, 3]})
        for i in range(self.T):
            df['d%d' % i] = X[:, 3 + i]

        return df
