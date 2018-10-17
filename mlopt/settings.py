import cvxpy as cp

# Define constants
TOL = 1e-06
PERTURB_TOL = 1e-05
BINDING_CONSTRAINTS_TOL = 1e-05
INFINITY = 1e15

# Define default solver
DEFAULT_SOLVER = cp.MOSEK

# Define learners
PYTORCH = "pytorch"
TENSORFLOW = "tensorflow"
OPTIMAL_TREE = "optimaltree"
DEFAULT_LEARNER = PYTORCH

# Learners settings
N_BEST = 3
