import cvxpy as cp

# Define constants
INFEAS_TOL = 1e-04
SUBOPT_TOL = 1e-04
TIGHT_CONSTRAINTS_TOL = 1e-4
DIVISION_TOL = 1e-8
FILTER_STRATEGIES_SAMPLES_FRACTION = 0.8
FILTER_SUBOPT = 2e-01

# Define default solver
DEFAULT_SOLVER = cp.GUROBI
#  DEFAULT_SOLVER = cp.CPLEX
#  DEFAULT_SOLVER = cp.MOSEK
#  DEFAULT_SOLVER = cp.ECOS

# Define learners
PYTORCH = "pytorch"
TENSORFLOW = "tensorflow"
OPTIMAL_TREE = "optimaltree"
DEFAULT_LEARNER = PYTORCH

# Learners settings
N_BEST = 3
FRAC_TRAIN = 0.9  # Fraction dividing training and validation

# Neural network default parameters
NET_TRAINING_PARAMS = {
    'learning_rate': [1e-04, 1e-03, 1e-02],
    'n_epochs': [20],
    'batch_size': [32],
    'n_layers': [2]
}

# Sampling
SAMPLING_TOL = 5e-03
