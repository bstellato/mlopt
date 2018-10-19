# Inventory example script
import numpy as np
import cvxpy as cp
import pandas as pd

# Development
import mlopt
#  import importlib
#  importlib.reload(mlopt)

'''
Define Inventory problem
'''
bin_vars = False

# Generate data
np.random.seed(1)
T = 2
M = 4.
K = 10.
radius = 3.0

# Define problem
x = cp.Variable(T+1)
u = cp.Variable(T)
y = cp.Variable(T+1)  # Auxiliary y = max(h * x, - p * x)

if bin_vars:
    v = cp.Variable(T, boolean=True)

# Define parameters
x_init = cp.Parameter(nonneg=True, name="x_init")
h = cp.Parameter(nonneg=True, name="h")
p = cp.Parameter(nonneg=True, name="p")
c = cp.Parameter(nonneg=True, name="c")
d = cp.Parameter(T, nonneg=True, name="d")

# Constaints
constraints = [x[0] == x_init]
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
cost = cp.sum(y) + c * cp.sum(u)
if bin_vars:
    cost += K * cp.sum(v)

# Define problem
problem = mlopt.Problem(cp.Problem(cp.Minimize(cost),
                                               constraints),
                                    name="inventory")

'''
Sample points
'''
# Operating point
theta_bar = np.array([
    4.,  # h
    6.,  # p
    3.5,  # c
    5.,  # x_0
    ])
theta_bar = np.concatenate((theta_bar, 5. * np.ones(T)))


def sample_inventory(theta_bar, radius, N=100):

    # Sample points from multivariate ball
    X = mlopt.uniform_sphere_sample(theta_bar, radius, N=N)

    df = pd.DataFrame({'h': X[:, 0],
                       'p': X[:, 1],
                       'c': X[:, 2],
                       'x_init': X[:, 3],
                       'd': X[:, 4:].tolist()})

    return df


'''
Train and solve
'''

# Training and testing data
n_train = 100
n_test = 10
theta_train = sample_inventory(theta_bar, radius, N=n_train)
theta_test = sample_inventory(theta_bar, radius, N=n_test)

# Encode training strategies
#  strategies = problem.solve_parametric(
#      theta_train,
#      message="Compute tight constraints for training set"
#  )[2]
results = problem.solve_parametric(
    theta_train,
    message="Compute tight constraints for training set"
)
y_train, encoding = mlopt.encode_strategies([r['strategy'] for r in results])

# Training
n_input = len(theta_bar)
n_classes = len(encoding)
#  with mlopt.TensorFlowNeuralNet(n_input, n_layers, n_classes) as learner:
with mlopt.PyTorchNeuralNet(n_input, n_classes) as learner:
#  with mlopt.OptimalTree() as learner:
    learner.train(theta_train, y_train)

    #  Testing
    results = mlopt.eval_performance(theta_test, learner, problem,
                                     encoding, k=3)

    mlopt.store(results, 'output/')
