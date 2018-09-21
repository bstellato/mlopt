# Inventory example script
import numpy as np
from problem import Inventory


# Development
import mlo
import importlib
importlib.reload(mlo)

#  def run_inventory():
# Generate data
np.random.seed(1)
T = 2
M = 4.
K = 10.
radius = 3.0

# Operating point
theta_bar = np.array([
    4.,  # h
    6.,  # p
    3.5,  # c
    5.,  # x_0
    ])
theta_bar = np.concatenate((theta_bar, 5. * np.ones(T)))

# Define problem
problem = Inventory(T, M, K, radius, bin_vars=False)

# Training and testing data
n_train = 2000
n_test = 100
theta_train = problem.sample(theta_bar, N=n_train)
theta_test = problem.sample(theta_bar, N=n_test)

# Encode training strategies
_, _, strategies = problem.solve_parametric(
    theta_train,
    message="Compute active constraints for training set"
)
y_train, enc2strategy = mlo.encode_strategies(strategies)

# Training
n_input = len(theta_bar)
n_layers = [15, 15]
n_classes = len(enc2strategy)
learner = mlo.NeuralNet(n_input, n_layers, n_classes)
learner.train(theta_train, y_train)

#  Testing
results = mlo.eval_performance(theta_test, learner, problem,
                               enc2strategy, k=3)
mlo.store(results, 'examples/output/inventory')
