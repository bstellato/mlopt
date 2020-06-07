from abc import ABC, abstractmethod
import numpy as np
import optuna
import mlopt.settings as stg


class Learner(ABC):
    """
    Optimization strategy learner

    Attributes
    ----------
    n_train : int
        Number of training samples.
    """

    @classmethod
    @abstractmethod
    def is_installed(cls):
        """Is learner installed?"""
        return NotImplemented

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
        return NotImplemented

    @abstractmethod
    def predict(self, X):
        """Predict strategies from data."""
        return NotImplemented

    @abstractmethod
    def save(self, file_name):
        """Save learner to file"""
        return NotImplemented

    @abstractmethod
    def load(self, file_name):
        """Load learner from file"""
        return NotImplemented

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

    def print_trial_stats(self, study):
        """TODO: Docstring for print_trial_stats.

        Args:
            arg1 (TODO): TODO

        Returns: TODO

        """
        best_params = study.best_trial.params

        pruned_trials = [t for t in study.trials
                         if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials
                           if t.state == optuna.trial.TrialState.COMPLETE]

        stg.logger.info("Study statistics: ")
        stg.logger.info("  Number of finished trials: %d" % len(study.trials))
        stg.logger.info("  Number of pruned trials: %d" % len(pruned_trials))
        stg.logger.info("  Number of complete trials: %d" %
                        len(complete_trials))

        stg.logger.info("Best loss value: %.4f" % study.best_trial.value)
        stg.logger.info("Best parameters")
        for key, value in best_params.items():
            print("    {}: {}".format(key, value))
