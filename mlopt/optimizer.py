from mlopt.problem import Problem
from mlopt.settings import DEFAULT_SOLVER, DEFAULT_LEARNER, TOL
from mlopt.learners import LEARNER_MAP
from mlopt.sampling import Sampler
from mlopt.strategy import encode_strategies
from mlopt.utils import n_features, accuracy
import pandas as pd
import numpy as np
import os
import shutil
import pickle as pkl
from tqdm import tqdm


class Optimizer(object):
    """
    Machine Learning Optimizer class.
    """

    def __init__(self,
                 objective, constraints,
                 perturb_problem=True,
                 name="problem",
                 **solver_options):
        """
        Inizialize optimizer.

        Parameters
        ----------
        objective : cvxpy objective
            Objective defined in CVXPY.
        constraints : cvxpy constraints
            Constraints defined in CVXPY.
        name : str
            Problem name.
        perturb_problem : bool, optional
            Perturb the cost of the optimization problem?
            This avoids degeneracy. Defaults to true.
        solver_options : dict, optional
            A dict of options for the internal solver.
        """
        self._problem = Problem(objective, constraints,
                                solver=DEFAULT_SOLVER,
                                perturb=perturb_problem,
                                **solver_options)
        self.name = name
        self._learner = None

    def sample(self, sampling_fn):
        """
        Sample parameters.
        """

        # Create sampler
        self._sampler = Sampler(self._problem, sampling_fn)

        # Sample parameters
        # TODO: CONTINUE FROM HERE
        # Move X_train and y_train computation here.

    def train(self, X, learner=DEFAULT_LEARNER, **learner_options):
        """
        Train optimizer using parameter X.

        Parameters
        ----------
        X : pandas dataframe or numpy array
            Data samples. Each row is a new sample points.
        learner : str
            Learner to use. Learners are defined in :mod:`mlopt.settings`
        learner_options : dict, optional
            A dict of options for the learner.
        """

        # Save training data internally
        self.X_train = X

        # Encode training strategies by solving
        # the problem for all the points
        results = self._problem.solve_parametric(X, message="Compute " +
                                                 "binding constraints " +
                                                 "for training set")
        train_strategies = [r['strategy'] for r in results]
        self.y_train, self.enc2strategy = encode_strategies(train_strategies)

        # Define learner
        self._learner = LEARNER_MAP[learner](n_input=n_features(self.X_train),
                                             n_classes=len(self.enc2strategy),
                                             **learner_options)

        # Train learner
        self._learner.train(self.X_train, self.y_train)

    def choose_best(self, strategies):
        """
        Choose best strategy between provided ones.
        """
        n_best = self._learner.options['n_best']

        # For each n_best classes get x, y, time and store the best one
        x = []
        time = []
        infeas = []
        cost = []

        for j in range(n_best):
            res = self._problem.solve_with_strategy(strategies[j])
            x.append(res['x'])
            time.append(res['time'])
            infeas.append(res['infeasibility'])
            cost.append(res['cost'])

        # Pick best class between k ones
        infeas = np.array(infeas)
        cost = np.array(cost)
        idx_filter = np.where(infeas <= TOL)[0]
        if len(idx_filter) > 0:
            # Case 1: Feasible points
            # -> Get solution with minimum cost
            #    between feasible ones
            idx_pick = idx_filter[np.argmin(cost[idx_filter])]
        else:
            # Case 2: No feasible points
            # -> Get solution with minimum infeasibility
            idx_pick = np.argmin(infeas)

        # Store values we are interested in
        result = {}
        result['x'] = x[idx_pick]
        result['time'] = np.sum(time)
        result['strategy'] = strategies[idx_pick]
        result['cost'] = cost[idx_pick]
        result['infeasibility'] = infeas[idx_pick]

        return result

    def solve(self, X,
              message="Predict optimal solution"):
        """
        Predict optimal solution given the parameters X.

        Parameters
        ----------
        X : pandas dataframe
            Data points.

        Returns
        -------
        list
            List of result dictionaries.
        """
        n_points = len(X)
        n_best = self._learner.options['n_best']

        # Define array of results to return
        results = []

        # Predict best n_best classes for all the points
        classes = self._learner.predict(X)

        for i in tqdm(range(n_points), desc=message):

            # Populate problem
            self._problem.populate(X.iloc[i, :])

            # Pick strategies from encoding
            strategies = [self.enc2strategy[classes[i, j]]
                          for j in range(n_best)]

            results.append(self.choose_best(strategies))

        return results

    def save(self, folder_name, delete_dir=False):
        """
        Save optimizer to a specific folder.

        The folder will be created if it does not exist.


        Parameters
        ----------
        folder_name : string
            Folder name where to store files.
        delete_dir : bool
            Delete folder if already existing?
        """

        if self._learner is None:
            raise ValueError("You cannot save the optimizer without " +
                             "training it before.")

        # Create directory
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        else:
            if not delete_dir:
                p = None
                while p not in ['y', 'n', 'N', '']:
                    p = input("Directory %s/ already exists. " % folder_name +
                              "Would you like to delete it? [y/N] ")
                if p == 'y':
                    shutil.rmtree(folder_name)
                    os.makedirs(folder_name)
            else:
                shutil.rmtree(folder_name)
                os.makedirs(folder_name)

        # Save learner
        self._learner.save(os.path.join(folder_name, "learner"))

        # Save optimizer
        optimizer = open(os.path.join(folder_name, "optimizer.pkl"), 'wb')
        file_dict = {'name': self.name,
                     'learner_name': self._learner.name,
                     'learner_options': self._learner.options,
                     'enc2strategy': self.enc2strategy,
                     'objective': self._problem.objective,
                     'constraints': self._problem.constraints}
        pkl.dump(file_dict, optimizer)
        optimizer.close()

    @classmethod
    def from_file(cls, folder_name):
        """
        Create optimizer from a specific folder.

        Parameters
        ----------
        folder_name : string
            Folder name where to read files from.
        """

        # Check if folder exists
        if not os.path.exists(folder_name):
            raise ValueError("Folder does not exist.")

        # Load optimizer
        optimizer_file_name = os.path.join(folder_name, "optimizer.pkl")
        if not optimizer_file_name:
            raise ValueError("Optimizer pkl file does not exist.")
        f = open(optimizer_file_name, "rb")
        optimizer_dict = pkl.load(f)
        f.close()

        # Create optimizer using loaded dict
        # Assume perturbation already happened
        optimizer = cls(optimizer_dict['objective'],
                        optimizer_dict['constraints'],
                        name=optimizer_dict['name'],
                        perturb_problem=False)

        # Assign strategies encoding
        optimizer.enc2strategy = optimizer_dict['enc2strategy']
        learner_name = optimizer_dict['learner_name']
        learner_options = optimizer_dict['learner_options']

        # Load learner
        optimizer._learner = \
            LEARNER_MAP[learner_name](n_input=optimizer._problem.n_parameters,
                                      n_classes=len(optimizer.enc2strategy),
                                      **learner_options)
        optimizer._learner.load(os.path.join(folder_name, "learner"))

        return optimizer

    def performance(self, theta):
        """
        Evaluate optimizer performance on data theta by comparing the
        solution to the optimal one.

        Parameters
        ----------
        theta : DataFrame
            Data to predict.

        Returns
        -------
        dict
            Results summarty.
        dict
            Detailed results summary.
        """

        print("Performance evaluation")
        # Get strategy for each point
        results_test = self._problem.solve_parametric(theta,
                                                      message="Compute " +
                                                      "binding constraints " +
                                                      "for test set")
        time_test = [r['time'] for r in results_test]
        strategy_test = [r['strategy'] for r in results_test]
        cost_test = [r['cost'] for r in results_test]

        # Get predicted strategy for each point
        results_pred = self.solve(theta,
                                  message="Predict binding constraints for " +
                                  "test set")
        time_pred = [r['time'] for r in results_pred]
        strategy_pred = [r['strategy'] for r in results_pred]
        cost_pred = [r['cost'] for r in results_pred]
        infeas = np.array([r['infeasibility'] for r in results_pred])

        n_test = len(theta)
        n_train = self._learner.n_train  # Number of training samples
        n_theta = n_features(theta)  # Number of parameters
        n_strategies = len(self.enc2strategy)  # Number of strategies

        # Compute comparative statistics
        time_comp = np.array([(1 - time_pred[i] / time_test[i])
                              for i in range(n_test)])
        subopt = np.array([(cost_pred[i] - cost_test[i])/(cost_test[i] + 1e-10)
                           for i in range(n_test)])

        # accuracy
        test_accuracy, idx_correct = accuracy(strategy_pred, strategy_test)

        # Create dataframes to return
        df = pd.DataFrame(
            {
                "problem": [self.name],
                "n_best": [self._learner.options['n_best']],
                "n_var": [self._problem.n_var],
                "nm_constr": [self._problem.n_constraints],
                "n_test": [n_test],
                "n_train": [n_train],
                "n_theta": [n_theta],
                "n_corect": [np.sum(idx_correct)],
                "n_strategies": [n_strategies],
                "accuracy": [test_accuracy],
                "n_infeas": [np.sum(infeas >= TOL)],
                "avg_infeas": [np.mean(infeas)],
                "avg_subopt": [np.mean(subopt[np.where(infeas <= TOL)[0]])],
                "max_infeas": [np.max(infeas)],
                "max_subopt": [np.max(subopt)],
                "avg_time_improv": [np.mean(time_comp)],
                "max_time_improv": [np.max(time_comp)],
            }
        )
        # Add radius info if problem has it.
        # TODO: We should remove it later
        #  try:
        #      df["radius"] = [self._problem.radius]
        #  except AttributeError:
        #      pass

        df_detail = pd.DataFrame(
            {
                "problem": [self.name] * n_test,
                "correct": idx_correct,
                "infeas": infeas,
                "subopt": subopt,
                "time_improvement": time_comp,
            }
        )

        return df, df_detail
