from mlopt.optimizer import Optimizer
from mlopt.settings import PYTORCH, OPTIMAL_TREE, XGBOOST
from mlopt.learners import installed_learners


# Suppress warnings
import warnings
from scikits.umfpack import UmfpackWarning
# Ignore umfpack warning when amtrix is singular
#  warnings.simplefilter('ignore', UmfpackWarning)
# TODO: Fixme, still prints
warnings.filterwarnings(
    action='ignore',
    category=RuntimeWarning,
    module='umfpack'
)
warnings.filterwarnings(
    action='ignore',
    category=UmfpackWarning,
    module='umfpack'
)

