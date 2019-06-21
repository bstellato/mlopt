from mlopt.learners.pytorch.pytorch import PyTorchNeuralNet
from mlopt.learners.optimal_tree.optimal_tree import OptimalTree
import mlopt.settings as s

LEARNER_MAP = {s.PYTORCH: PyTorchNeuralNet,
               s.OPTIMAL_TREE: OptimalTree}
