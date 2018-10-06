from abc import ABC, abstractmethod
from ..constants import TOL
import numpy as np
from tqdm import tqdm


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
        """Learn predictor form data"""

    @abstractmethod
    def predict(self, X):
        """Predict strategy from data"""

    @abstractmethod
    def __enter__(self):
        """Enter for context manager"""

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        """Exit for context manager"""

    def pandas2array(self, X):
        """
        Unroll dataframe elements to construct 2d array in case of
        cells containing tuples.
        """

        # get number of datapoints
        n_data = len(X)
        # Get dimensions by inspecting first row
        n = 0
        for c in X.columns.values:

            if isinstance(X[c][0], list):
                # If list add length
                n += len(X[c][0])
            else:
                # If number add 1
                n += 1

        # Allocate full vector
        X_new = np.empty((0, n))
        #  X_new = np.array([], shape=(0, n))

        # Unroll
        # TODO: Speedup this process
        for i in range(n_data):
            x_temp = np.array([])
            x_data = X.iloc[i, :].values
            for i in x_data:
                if isinstance(i, list):
                    x_temp = np.concatenate((x_temp, np.array(i)))
                else:
                    x_temp = np.append(x_temp, i)

            X_new = np.vstack((X_new, x_temp))

        return X_new

    def predict_best_points(self, X, problem, k, enc2strategy,
                            message="Predict active constraints"):
        """
        Predict best points by picking the best values when solving the problem
        """
        n_points = len(X)

        # Data to return for each point
        strategy = []
        x = []
        time = []

        # Predict best classes for all the points
        classes = self.predict_best(X, k=k)

        for i in tqdm(range(n_points), desc=message):

            # Populate problem
            problem.populate(X.iloc[i, :])

            # Encode strategies
            strategy_classes = [enc2strategy[classes[i, j]] for j in range(k)]

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
            infeas_temp = np.array(infeas_temp)
            cost_temp = np.array(cost_temp)
            idx_filter = np.where(infeas_temp <= TOL)[0]
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
