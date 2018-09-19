from abc import ABC


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

    @ABC.abstractmethod
    def train(self, X, y):
        """Learn predictor form data"""

    @ABC.abstractmethod
    def predict(self, X):
        """Predict strategy from data"""

    # TODO: Add enc2strategy as an attribute

    #  def predict_best(X, k)


