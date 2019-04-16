from mlopt.optimizer import Optimizer
from mlopt.settings import PYTORCH, OPTIMAL_TREE


# Suppress warnings
import warnings
from scikits.umfpack import UmfpackWarning
warnings.simplefilter(action='ignore', category=UmfpackWarning)

