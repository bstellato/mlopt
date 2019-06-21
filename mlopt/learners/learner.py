from abc import ABC, abstractmethod
import numpy as np


class Learner(ABC):
    """
    Optimization strategy learner

    Attributes
    ----------
    n_train : int
        Number of training samples.
    """
    @property
    def n_train(self):
        """Number of training samples."""
        return self._n_train

    @n_train.setter
    def n_train(self, value):
        self._n_train = value

    @abstractmethod
    def train(self, X, y):
        """Learn predictor form data."""

    @abstractmethod
    def predict(self, X):
        """Predict strategies from data."""

    @abstractmethod
    def save(self, file_name):
        """Save learner to file"""

    @abstractmethod
    def load(self, file_name):
        """Load learner from file"""

    def pick_best_class(self, y, n_best=None):
        """
        Sort predictions and pick best points.

        Use n_best classes to choose classes that
        are most likely.
        """
        n_points = y.shape[0]
        n_best = n_best if (n_best is not None) else self.options['n_best']

        # Sort probabilities
        idx_probs = np.empty((n_points, n_best), dtype='int')
        for i in range(n_points):
            # Get best k indices
            # NB. Argsort sorts in reverse mode
            idx_probs[i, :] = np.argsort(y[i, :])[-n_best:]

        return idx_probs
