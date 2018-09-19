from abc import ABC
from ..constants import TOL
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

    @ABC.abstractmethod
    def train(self, X, y):
        """Learn predictor form data"""

    @ABC.abstractmethod
    def predict(self, X):
        """Predict strategy from data"""

    def predict_best_points(self, X, problem, k, enc2strategy):
        """
        Predict best points by picking the best values when solving the problem
        """
        n_points = len(X)

        # Data to return for each point
        strategy = []
        x = []
        time = []

        for i in range(n_points):

            # Predict best k classes
            classes = self.predict_best(X.iloc[i, :], k=k)

            # Populate problem
            problem.populate(X.iloc[i, :])

            # Encode strategies
            strategy_classes = [enc2strategy[classes[j]] for j in range(k)]

            # For each k classes get x, y, time and store the best one
            x_temp = []
            time_temp = []
            infeas_temp = []
            cost_temp = []

            for j in range(k):
                sol = problem.solve_with_strategy(strategy_classes[j])
                x_temp.append(sol[0])
                time_temp.append(sol[1])

                # Compute infeasibility
                infeas_temp.append(problem.infeasibility(x_temp[-1]))

                # Compute cost
                cost_temp.append(problem.cost(x_temp[-1]))

                # Pick best class between k ones
                idx_filter = np.where(infeas_temp <= TOL)
                if len(idx_filter) > 0:
                    # Case 1: Feasible points
                    # -> Get solution with minimum cost
                    #    between feasible ones
                    idx_pick = idx_filter[np.argmin(cost_temp[idx_filter])]
                else:
                    # Case 2: No feasible points
                    # -> Get solution with minimum infeasibility
                    idx_pick = np.argmin(infeas_temp)

                # Store values we are interested in
                x.append(x_temp[idx_pick])
                time.append(np.sum(time_temp))
                strategy.append(strategy_classes[idx_pick])

            # Return x, time and strategy for all the points
            return x, time, strategy























