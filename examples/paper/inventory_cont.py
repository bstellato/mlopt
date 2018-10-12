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
T = 5
M = 2.
K = 1.
h = 1.
c = 1.
p = 1.
x_init = 2.

# Define problem
x = cp.Variable(T+1)
u = cp.Variable(T)
#  y = cp.Variable(T+1)  # Auxiliary y = max(h * x, - p * x)

# Define parameters
d = cp.Parameter(T, nonneg=True, name="d")

# Constaints
constraints = [x[0] == x_init]
for t in range(T):
    constraints += [x[t+1] == x[t] + u[t] - d[t]]
constraints += [u >= 0, u <= M]

# Objective
#  cost = cp.sum(y) + c * cp.sum(u)
cost = cp.sum(cp.maximum(h * x, -p * x)) + c * cp.sum(u)

# Define problem
cvxpy_problem = cp.Problem(cp.Minimize(cost), constraints)
problem = mlopt.OptimizationProblem(cvxpy_problem,
                                    name="inventory")

'''
Sample points
'''
# Average request
theta_bar = 3. * np.ones(T)
radius = 2.0

def sample_inventory(theta_bar, radius, N=100):

    # Sample points from multivariate ball
    X = mlopt.uniform_sphere_sample(theta_bar, radius, N=N)

    df = pd.DataFrame({'d': X.tolist()})

    return df


'''
Train and solve
'''

# Training and testing data
n_train = 10
#  n_test = 50
theta_train = sample_inventory(theta_bar, radius, N=n_train)
#  theta_test = sample_inventory(theta_bar, radius, N=n_test)

# Encode training strategies
results = problem.solve_parametric(
    theta_train,
    message="Compute binding constraints for training set"
)
y_train, enc2strategy = mlopt.encode_strategies([r['strategy'] for r in results])

#  # Training
#  n_input = len(theta_bar)
#  n_classes = len(enc2strategy)
#  #  with mlopt.TensorFlowNeuralNet(n_input, n_layers, n_classes) as learner:
#  with mlopt.PyTorchNeuralNet(n_input, n_classes) as learner:
#  #  with mlopt.OptimalTree() as learner:
#      learner.train(theta_train, y_train)
#
#      #  Testing
#      results = mlopt.eval_performance(theta_test, learner, problem,
#                                       enc2strategy, k=3)
#
#      mlopt.store(results, 'output/')
#
