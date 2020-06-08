import cvxpy as cp

# Logger
import logging
import sys
LOGGER_NAME = 'mlopt'
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)

# Stdout handler
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_formatter = logging.Formatter('%(message)s')
logger.addHandler(stdout_handler)
logger.propagate = False   # Disable double logging

# Add file handler
#  file_handler = logging.FileHandler('mlopt.log')
#  file_handler.setLevel(logging.INFO)
#  logger.addHandler(file_handler)


# Define constants
INFEAS_TOL = 1e-04
SUBOPT_TOL = 1e-04
TIGHT_CONSTRAINTS_TOL = 1e-4
DIVISION_TOL = 1e-8

# Define default solver
DEFAULT_SOLVER = cp.GUROBI
DEFAULT_SOLVER_OPTIONS = {'Method': 1}  # Dual simplex
#  DEFAULT_SOLVER = cp.CPLEX
#  DEFAULT_SOLVER = cp.MOSEK
#  DEFAULT_SOLVER = cp.ECOS

# Define learners
PYTORCH = "pytorch"
TENSORFLOW = "tensorflow"
OPTIMAL_TREE = "optimaltree"
XGBOOST = "xgboost"
DEFAULT_LEARNER = XGBOOST

# Learners settings
N_BEST = 10
N_TRAIN_TRIALS = 100
FRAC_TRAIN = 0.9  # Fraction dividing training and validation


# Sampling
SAMPLING_TOL = 5e-03

# Filtering
FILTER_STRATEGIES = True
FILTER_STRATEGIES_SAMPLES_FRACTION = 0.98
FILTER_SUBOPT = 2e-01
