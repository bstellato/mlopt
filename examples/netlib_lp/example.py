import numpy as np
import os
from glob import glob
#  from problem import NetlibLP

# Development
import mlopt
import importlib
importlib.reload(mlopt)

# Get problems by file size
lp_data_dir = os.path.join("benchmarks", "netlib_lp", "lp_data", "mat")
files = np.array(glob(os.path.join(lp_data_dir, "*.mat")))
file_sizes = np.array([os.path.getsize(f) for f in files])
files = files[np.argsort(file_sizes)]  # Files ordered by size


# Take only a few files
files = files[2]  # Afiro

#  # Generate data
#  np.random.seed(1)
#  T = 10
#  M = 4.
#  K = 10.
#  radius = 3.0
#
#  # Operating point
#  theta_bar = np.array([
#      4.,  # h
#      6.,  # p
#      3.5,  # c
#      5.,  # x_0
#      ])
#  theta_bar = np.concatenate((theta_bar, 5. * np.ones(T)))
#
#  # Define problem
#  problem = Inventory(T, M, K, radius, bin_vars=True)
#
#  # Training and testing data
#  n_train = 5000
#  n_test = 100
#  theta_train = problem.sample(theta_bar, N=n_train)
#  theta_test = problem.sample(theta_bar, N=n_test)
#
#  # Encode training strategies
#  _, _, strategies = problem.solve_parametric(
#      theta_train,
#      message="Compute tight constraints for training set"
#  )
#  y_train, encoding = mlopt.encode_strategies(strategies)
#
#  # Training
#  n_input = len(theta_bar)
#  n_layers = [15, 15]
#  n_classes = len(encoding)
#  with mlopt.NeuralNet(n_input, n_layers, n_classes) as learner:
#      learner.train(theta_train, y_train)
#
#      #  Testing
#      results = mlopt.eval_performance(theta_test, learner, problem,
#                                     encoding, k=3)
#      mlopt.store(results, 'examples/output/inventory')
