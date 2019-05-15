from mlopt.optimizer import Optimizer
from mlopt.settings import PYTORCH, OPTIMAL_TREE


# Suppress warnings
import warnings
from scikits.umfpack import UmfpackWarning
# Ignore umfpack warning when amtrix is singular
warnings.simplefilter('ignore', UmfpackWarning)
warnings.filterwarnings(
    action='ignore',
    category=RuntimeWarning,
    module='scikits.umfpack'
)

