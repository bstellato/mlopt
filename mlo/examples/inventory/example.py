# Inventory example script
import numpy as np
from .problem import Inventory
from ...strategy import encode_strategies
from ...learners.neural_net import NeuralNet
from ...performance import eval_performance, store

# Generate data
T = 10
M = 3.
K = 1.
radius = 1.0

# Operating point
theta_bar = np.array([
    2.,  # h
    2.,  # p
    5.,  # c
    1.,  # x_0
    ])

theta_bar = np.append((theta_bar, np.ones(T)))

# Define problem
problem = Inventory(T, M, K, radius)

# Training and testing data
n_train = 5000
n_test = 100
theta_train = problem.sample(theta_bar, N=n_train)
theta_test = problem.sample(theta_bar, N=n_test)

# Encode training strategies
_, _, strategies = problem.solve_parametric(theta_train)
y_train, enc2strategy = encode_strategies(strategies)

# Training
n_layers = 2
n_classes = len(enc2strategy)
learner = NeuralNet(n_layers, n_classes)
learner.train(theta_train, y_train)

# Testing
results = eval_performance(theta_test, learner, problem,
                           enc2strategy, k=3)
store(results)
