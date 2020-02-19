import cvxpy as cp

# Logger
import logging
import sys
LOGGER_NAME = 'mlopt'
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# DEFAULT settings
FILTER_STRATEGIES = False

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
DEFAULT_LEARNER = PYTORCH

# Learners settings
N_BEST = 10
FRAC_TRAIN = 0.9  # Fraction dividing training and validation

# Optimal Trees default parameters
OPTIMAL_TREE_TRAINING_PARAMS = {
    'max_depth': [5, 10, 15],
    'minbucket': [1, 5, 10],
    'hyperplanes': False,
}

# Neural network default parameters
NET_TRAINING_PARAMS = {
    'learning_rate': [1e-04, 1e-03, 1e-02],
    'n_epochs': [20],
    'batch_size': [32],
    # 'n_layers': [5, 7, 10]
}

# Sampling
SAMPLING_TOL = 5e-03

# Filtering
FILTER_STRATEGIES_SAMPLES_FRACTION = 0.8
FILTER_SUBOPT = 2e-01
