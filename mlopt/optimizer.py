from mlopt.problem import Problem
from mlopt.settings import DEFAULT_SOLVER, DEFAULT_LEARNER, INFEAS_TOL
from mlopt.learners import LEARNER_MAP
from mlopt.sampling import Sampler
from mlopt.strategy import encode_strategies
from mlopt.utils import n_features, accuracy, suboptimality
import cvxpy.settings as cps
import pandas as pd
import numpy as np
import os
from glob import glob
import tempfile
import tarfile
import pickle as pkl
from tqdm import tqdm


class Optimizer(object):
    """
    Machine Learning Optimizer class.
    """

    def __init__(self,
                 objective, constraints,
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
        solver_options : dict, optional
            A dict of options for the internal solver.
        """
        self._problem = Problem(objective, constraints,
                                solver=DEFAULT_SOLVER,
                                **solver_options)
        self.name = name
        self._learner = None
        self.encoding = None
        self.X_train = None
        self.y_train = None

    @property
    def n_parameters(self):
        """Number of parameters."""
        return self._problem.n_parameters

    def sample(self, sampling_fn):
        """
        Sample parameters.
        """

        # Create sampler
        self._sampler = Sampler(self._problem, sampling_fn)

        # Sample parameters
        self.X_train, self.y_train, self.encoding = self._sampler.sample()

    def save_data(self, file_name, delete_existing=False):
        """
        Save data points to file.


        Avoids the need to recompute data.

        Parameters
        ----------
        file_name : string
            File name of the compressed optimizer.
        delete_existing : bool, optional
            Delete existing file with the same name?
            Defaults to False.
        """
        # Check if file already exists
        if os.path.isfile(file_name):
            if not delete_existing:
                p = None
                while p not in ['y', 'n', 'N', '']:
                    p = input("File %s already exists. " % file_name +
                              "Would you like to delete it? [y/N] ")
                if p == 'y':
                    os.remove(file_name)
                else:
                    return
            else:
                os.remove(file_name)

        if (self.X_train is None) or \
            (self.y_train is None) or \
                (self.encoding is None):
            raise ValueError("You need to get the strategies " +
                             "from the data first by training the model.")

        # Save to file
        with open(file_name, 'wb') \
                as data:
            data_dict = {'X_train': self.X_train,
                         'y_train': self.y_train,
                         'problem': self._problem,
                         'encoding': self.encoding}
            pkl.dump(data_dict, data)

    def load_data(self, file_name):
        """
        Load pickled data from file name.

        Parameters
        ----------
        file_name : string
            File name of the data.
        """

        # Check if file exists
        if not os.path.isfile(file_name):
            raise ValueError("File %s does not exist." % file_name)

        # Load optimizer
        with open(file_name, "rb") as f:
            data_dict = pkl.load(f)

        # Store data internally
        self.X_train = data_dict['X_train']
        self.y_train = data_dict['y_train']
        self._problem = data_dict['problem']
        self.encoding = data_dict['encoding']

    def train(self, X=None, sampling_fn=None,
              parallel=True,
              learner=DEFAULT_LEARNER,
              **learner_options):
        """
        Train optimizer using parameter X.

        This function needs one argument between data points X
        or sampling function sampling_fn. It will raise an error
        otherwise because there is no way to sample data.

        Parameters
        ----------
        X : pandas dataframe or numpy array, optional
            Data samples. Each row is a new sample points.
        sampling_fn : function, optional
            Function to sample data taking one argument being
            the number of data points to be sampled and returning
            a structure of the same type as X.
        parallel : bool
            Perform training in parallel.
        learner : str
            Learner to use. Learners are defined in :mod:`mlopt.settings`
        learner_options : dict, optional
            A dict of options for the learner.
        """

        # Check if data is passed, otherwise train
        if X is not None:
            self.X_train = X
        elif sampling_fn is not None:
            # Create X_train, y_train and encoding from
            # sampling function
            self.sample(sampling_fn)
        else:
            # Use samples already provided
            if self.encoding is None:
                # Encode training strategies by solving
                # the problem for all the points
                results = self._problem.solve_parametric(X,
                                                         parallel=parallel,
                                                         message="Compute " +
                                                         "tight constraints " +
                                                         "for training set")
                train_strategies = [r['strategy'] for r in results]

                # Check if the problems are solvable
                for r in results:
                    assert r['status'] in cps.SOLUTION_PRESENT, \
                        "The training points must be feasible"

                # Encode strategies
                self.y_train, self.encoding = encode_strategies(train_strategies)

        # Define learner
        self._learner = LEARNER_MAP[learner](n_input=n_features(self.X_train),
                                             n_classes=len(self.encoding),
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
        idx_filter = np.where(infeas <= INFEAS_TOL)[0]
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
            strategies = [self.encoding[classes[i, j]]
                          for j in range(n_best)]

            results.append(self.choose_best(strategies))

        return results

    def save(self, file_name, delete_existing=False):
        """
        Save optimizer to a specific tar.gz file.

        Parameters
        ----------
        file_name : string
            File name of the compressed optimizer.
        delete_existing : bool, optional
            Delete existing file with the same name?
            Defaults to False.
        """
        if self._learner is None:
            raise ValueError("You cannot save the optimizer without " +
                             "training it before.")

        # Add .tar.gz if the file has no extension
        if not file_name.endswith('.tar.gz'):
            file_name += ".tar.gz"

        # Check if file already exists
        if os.path.isfile(file_name):
            if not delete_existing:
                p = None
                while p not in ['y', 'n', 'N', '']:
                    p = input("File %s already exists. " % file_name +
                              "Would you like to delete it? [y/N] ")
                if p == 'y':
                    os.remove(file_name)
                else:
                    return
            else:
                os.remove(file_name)

        # Create temporary directory to create the archive
        # and store relevant files
        with tempfile.TemporaryDirectory() as tmpdir:

            # Save learner
            self._learner.save(os.path.join(tmpdir, "learner"))

            # Save optimizer
            with open(os.path.join(tmpdir, "optimizer.pkl"), 'wb') \
                    as optimizer:
                file_dict = {'name': self.name,
                             'learner_name': self._learner.name,
                             'learner_options': self._learner.options,
                             'encoding': self.encoding,
                             'objective': self._problem.objective,
                             'constraints': self._problem.constraints}
                pkl.dump(file_dict, optimizer)

            # Create archive with the files
            tar = tarfile.open(file_name, "w:gz")
            for f in glob(os.path.join(tmpdir, "*")):
                tar.add(f, os.path.basename(f))
            tar.close()

    @classmethod
    def from_file(cls, file_name):
        """
        Create optimizer from a specific compressed tar.gz file.

        Parameters
        ----------
        file_name : string
            File name of the exported optimizer.
        """

        # Add .tar.gz if the file has no extension
        if not file_name.endswith('.tar.gz'):
            file_name += ".tar.gz"

        # Check if file exists
        if not os.path.isfile(file_name):
            raise ValueError("File %s does not exist." % file_name)

        # Extract file to temporary directory and read it
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(file_name) as tar:
                tar.extractall(path=tmpdir)

            # Load optimizer
            optimizer_file_name = os.path.join(tmpdir, "optimizer.pkl")
            if not optimizer_file_name:
                raise ValueError("Optimizer pkl file does not exist.")
            f = open(optimizer_file_name, "rb")
            optimizer_dict = pkl.load(f)
            f.close()

            # Create optimizer using loaded dict
            optimizer = cls(optimizer_dict['objective'],
                            optimizer_dict['constraints'],
                            name=optimizer_dict['name'])

            # Assign strategies encoding
            optimizer.encoding = optimizer_dict['encoding']
            learner_name = optimizer_dict['learner_name']
            learner_options = optimizer_dict['learner_options']

            # Load learner
            optimizer._learner = \
                LEARNER_MAP[learner_name](n_input=optimizer.n_parameters,
                                          n_classes=len(optimizer.encoding),
                                          **learner_options)
            optimizer._learner.load(os.path.join(tmpdir, "learner"))

        return optimizer

    def performance(self, theta, parallel=True):
        """
        Evaluate optimizer performance on data theta by comparing the
        solution to the optimal one.

        Parameters
        ----------
        theta : DataFrame
            Data to predict.
        parallel : bool, optional
            Solve problems in parallel? Defaults to True.

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
                                                      parallel=parallel,
                                                      message="Compute " +
                                                      "tight constraints " +
                                                      "for test set")
        time_test = [r['time'] for r in results_test]
        #  strategy_test = [r['strategy'] for r in results_test]
        cost_test = [r['cost'] for r in results_test]

        # Get predicted strategy for each point
        results_pred = self.solve(theta,
                                  message="Predict tight constraints for " +
                                  "test set")
        time_pred = [r['time'] for r in results_pred]
        #  strategy_pred = [r['strategy'] for r in results_pred]
        cost_pred = [r['cost'] for r in results_pred]
        infeas = np.array([r['infeasibility'] for r in results_pred])

        n_test = len(theta)
        n_train = self._learner.n_train  # Number of training samples
        n_theta = n_features(theta)  # Number of parameters
        n_strategies = len(self.encoding)  # Number of strategies

        # Compute comparative statistics
        time_comp = np.array([(1 - time_pred[i] / time_test[i])
                              for i in range(n_test)])
        subopt = np.array([suboptimality(cost_pred[i], cost_test[i])
                           for i in range(n_test)])

        # accuracy
        test_accuracy, idx_correct = accuracy(results_pred, results_test)

        # Create dataframes to return
        df = pd.DataFrame(
            {
                "problem": [self.name],
                "learner": [self._learner.name],
                "n_best": [self._learner.options['n_best']],
                "n_var": [self._problem.n_var],
                "n_constr": [self._problem.n_constraints],
                "n_test": [n_test],
                "n_train": [n_train],
                "n_theta": [n_theta],
                "n_correct": [np.sum(idx_correct)],
                "n_strategies": [n_strategies],
                "accuracy": [100 * test_accuracy],
                "n_infeas": [np.sum(infeas >= INFEAS_TOL)],
                "avg_infeas": [np.mean(infeas)],
                "avg_subopt": [np.mean(subopt[np.where(infeas <=
                                                       INFEAS_TOL)[0]])],
                "max_infeas": [np.max(infeas)],
                "max_subopt": [np.max(subopt)],
                "avg_time_improv": [100 * np.mean(time_comp)],
                "max_time_improv": [100 * np.max(time_comp)],
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
                "learner": [self._learner.name] * n_test,
                "correct": idx_correct,
                "infeas": infeas,
                "subopt": subopt,
                "time_improvement": time_comp,
            }
        )

        return df, df_detail
