import unittest
import numpy as np
from mlopt.problem import OptimizationProblem
from mlopt.sampling import uniform_sphere_sample
import pandas as pd
import cvxpy as cp

# Generate data
np.random.seed(1)
T = 5
M = 2.
h = 1.
c = 1.
p = 1.
x_init = 2.
radius = 3.

# Define problem
x = cp.Variable(T+1)
u = cp.Variable(T)

# Define parameter and sampling points
d = cp.Parameter(T, nonneg=True, name="d")
d_bar = 3. * np.ones(T)
X_d = uniform_sphere_sample(d_bar, radius, N=100)
df = pd.DataFrame({'d': X_d.tolist()})

# Constaints
constraints = [x[0] == x_init]
for t in range(T):
    constraints += [x[t+1] == x[t] + u[t] - d[t]]
constraints += [u >= 0, u <= M]

# Objective
# TODO: If you remove that part it reports a crappy solution
cost = cp.sum(cp.maximum(h * x, -p * x)) + c * cp.sum(u) + \
    1e-08 * cp.sum_squares(u)

# Define problem
cvxpy_problem = cp.Problem(cp.Minimize(cost), constraints)
problem = OptimizationProblem(cvxpy_problem)

# Solve for all theta in serial
results_serial = problem.solve_parametric(df,
                                          parallel=False)

# Solve for all theta in parallel
results_parallel = problem.solve_parametric(df,
                                            parallel=True)

# Assert all results match
import ipdb
ipdb.set_trace()
