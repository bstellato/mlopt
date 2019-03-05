import cvxpy as cp

# Define constants
INFEAS_TOL = 1e-04
SUBOPT_TOL = 1e-04
TIGHT_CONSTRAINTS_TOL = 1e-04
DIVISION_TOL = 1e-6

# Define default solver
DEFAULT_SOLVER = cp.GUROBI
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

# Sampling
SAMPLING_TOL = 5e-03
