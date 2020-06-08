import pandas as pd
from mlopt.problem import Problem
import mlopt.settings as stg
from mlopt.learners import LEARNER_MAP, installed_learners
from mlopt.sampling import Sampler
from mlopt.strategy import encode_strategies
from mlopt.filter import Filter
import mlopt.error as e
from mlopt.utils import n_features, accuracy, suboptimality
import mlopt.utils as u
from mlopt.kkt import create_kkt_matrix, factorize_kkt_matrix
from mlopt.utils import pandas2array
from cvxpy import Minimize, Maximize
import numpy as np
import os
from glob import glob
import tempfile
import tarfile
import pickle as pkl
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from time import time


class Optimizer(object):
    """
    Machine Learning Optimizer class.
    """

    def __init__(self,
                 cvxpy_problem,
                 name="problem",
                 log_level=None,
                 parallel=True,
                 tight_constraints=True,
                 **solver_options):
        """
        Inizialize optimizer.

        Parameters
        ----------
        problem : cvxpy.Problem
            Problem in CVXPY format.
        name : str
            Problem name.
        solver_options : dict, optional
            A dict of options for the internal solver.
        """

        if log_level is not None:
            stg.logger.setLevel(log_level)

        self._problem = Problem(cvxpy_problem,
                                solver=stg.DEFAULT_SOLVER,
                                tight_constraints=tight_constraints,
                                **solver_options)
        self._solver_cache = None
        self.name = name
        self._learner = None
        self.encoding = None
        self.X_train = None
        self.y_train = None

    @property
    def n_strategies(self):
        """Number of strategies."""
        if self.encoding is None:
            e.value_error("Model has been trained yet to " +
                    "return the number of strategies.")

        return len(self.encoding)

    def variables(self):
        """Problem variables."""
        return self._problem.variables()

    @property
    def parameters(self):
        """Problem parameters."""
        return self._problem.parameters

    @property
    def n_parameters(self):
        """Number of parameters."""
        return self._problem.n_parameters

    def samples_present(self):
        """Check if samples have been generated."""
        return (self.X_train is not None) and \
            (self.y_train is not None) and \
            (self.encoding is not None)

    def sample(self, sampling_fn, parallel=False):
        """
        Sample parameters.
        """

        # Create sampler
        self._sampler = Sampler(self._problem, sampling_fn)

        # Sample parameters
        self.X_train, self.y_train, self.obj_train, self.encoding = \
            self._sampler.sample(parallel=parallel)

    def save_training_data(self, file_name, delete_existing=False):
        """
        Save training data to file.


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

        if not self.samples_present():
            e.value_error("You need to get the strategies " +
                          "from the data first by training the model.")

        # Save to file
        with open(file_name, 'wb') \
                as data:
            data_dict = {'X_train': self.X_train,
                         'y_train': self.y_train,
                         'obj_train': self.obj_train,
                         '_problem': self._problem,
                         'encoding': self.encoding}

            # if hasattr(self, '_solver_cache'):
            #     data_dict['_solver_cache'] = self._solver_cache

            # Store strategy filter
            if hasattr(self, '_filter'):
                data_dict['_filter'] = self._filter

            pkl.dump(data_dict, data)

    def load_training_data(self, file_name):
        """
        Load pickled training data from file name.

        Parameters
        ----------
        file_name : string
            File name of the data.
        """

        # Check if file exists
        if not os.path.isfile(file_name):
            e.value_error("File %s does not exist." % file_name)

        # Load optimizer
        with open(file_name, "rb") as f:
            data_dict = pkl.load(f)

        # Store data internally
        self.X_train = data_dict['X_train']
        self.y_train = data_dict['y_train']
        self.obj_train = data_dict['obj_train']
        self._problem = data_dict['_problem']
        self.encoding = data_dict['encoding']

        stg.logger.info("Loaded %d points with %d strategies" %
                        (len(self.y_train), len(self.encoding)))

        if ('_solver_cache' in data_dict):
            self._solver_cache = data_dict['_solver_cache']

        # Full strategies backup after filtering
        if ('_filter' in data_dict):
            self._filter = data_dict['_filter']

        # Compute Good turing estimates
        self._sampler = Sampler(self._problem, n_samples=len(self.X_train))
        self._sampler.compute_good_turing(self.y_train)

    def get_samples(self, X=None, sampling_fn=None,
                    parallel=True,
                    filter_strategies=stg.FILTER_STRATEGIES):
        """Get samples either from data or from sampling function"""

        # Assert we have data to train or already trained
        if X is None and sampling_fn is None and not self.samples_present():
            e.value_error("Not enough arguments to train the model")

        if X is not None and sampling_fn is not None:
            e.value_error("You can pass only one value between X "
                          "and sampling_fn")

        # Check if data is passed, otherwise train
        #  if (X is not None) and not self.samples_present():
        if X is not None:
            stg.logger.info("Use new data")
            self.X_train = X
            self.y_train = None
            self.encoding = None

            # Encode training strategies by solving
            # the problem for all the points
            results = self._problem.solve_parametric(X,
                                                     parallel=parallel,
                                                     message="Compute " +
                                                     "tight constraints " +
                                                     "for training set")

            not_feasible_points = {i: x for i, x in enumerate(results)
                                   if np.isnan(x['x']).any()}
            if not_feasible_points:
                e.value_error("Infeasible points found. Number of infeasible "
                              "points %d" % len(not_feasible_points))

            self.obj_train = [r['cost'] for r in results]
            train_strategies = [r['strategy'] for r in results]

            # Check if the problems are solvable
            #  for r in results:
            #      assert r['status'] in cps.SOLUTION_PRESENT, \
            #          "The training points must be feasible"

            # Encode strategies
            self.y_train, self.encoding = \
                encode_strategies(train_strategies)

            # Compute Good turing estimates
            self._sampler = Sampler(self._problem, n_samples=len(self.X_train))
            self._sampler.compute_good_turing(self.y_train)

            # Condense strategies
            if filter_strategies:
                self.filter_strategies(parallel=parallel)

        elif sampling_fn is not None and not self.samples_present():
            stg.logger.info("Use iterative sampling")
            # Create X_train, y_train and encoding from
            # sampling function
            self.sample(sampling_fn, parallel=parallel)

            # Condense strategies
            if filter_strategies:
                self.filter_strategies(parallel=parallel)

        # Add factorization faching if
        # 1. Problem is MIQP
        # 2. Parameters do not enter in matrices
        if self._problem.is_qp() and \
                (self._solver_cache is None) and \
                not self._problem.parameters_in_matrices:
            self.cache_factors()

    def filter_strategies(self, parallel=True):
        # Store full non filtered strategies

        self.encoding_full = self.encoding
        self.y_train_full = self.y_train

        # Define strategies filter (not run it yet)
        self._filter = Filter(X_train=self.X_train,
                              y_train=self.y_train,
                              obj_train=self.obj_train,
                              encoding=self.encoding,
                              problem=self._problem)
        self.y_train, self.encoding = \
            self._filter.filter(parallel=parallel)

    def train(self, X=None,
              sampling_fn=None,
              parallel=True,
              learner=stg.DEFAULT_LEARNER,
              filter_strategies=stg.FILTER_STRATEGIES,
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

        # Get training samples
        self.get_samples(X, sampling_fn,
                         parallel=parallel,
                         filter_strategies=filter_strategies)

        # Define learner
        if learner not in installed_learners():
            e.value_error("Learner specified not installed. "
                          "Available learners are: %s" % installed_learners())
        self._learner = LEARNER_MAP[learner](n_input=n_features(self.X_train),
                                             n_classes=len(self.encoding),
                                             **learner_options)

        # Train learner
        self._learner.train(pandas2array(self.X_train),
                            self.y_train)

    def cache_factors(self):
        """Cache linear system solver factorizations"""

        self._solver_cache = []
        stg.logger.info("Caching KKT solver factors for each strategy ")
        for strategy_idx in tqdm(range(self.n_strategies)):

            # Get a parameter giving that strategy
            strategy = self.encoding[strategy_idx]
            idx_param = np.where(self.y_train == strategy_idx)[0]
            theta = self.X_train.iloc[idx_param[0]]

            # Populate
            self._problem.populate(theta)

            # Get problem data
            data, inverse_data, solving_chain = \
                self._problem._get_problem_data()

            # Apply strategy
            strategy.apply(data, inverse_data[-1])

            # Old
            #  self._problem.populate(theta)
            #
            #  self._problem._relax_disc_var()
            #
            #  reduced_problem = \
            #      self._problem._construct_reduced_problem(strategy)
            #
            #  data, full_chain, inv_data = \
            #      reduced_problem.get_problem_data(solver=KKT)

            # Get KKT matrix
            KKT_mat = create_kkt_matrix(data)
            solve_kkt = factorize_kkt_matrix(KKT_mat)

            cache = {}
            cache['factors'] = solve_kkt
            #  cache['inverse_data'] = inverse_data
            #  cache['chain'] = solving_chain

            self._solver_cache += [cache]

    def choose_best(self, labels, parallel=False, use_cache=True):
        """
        Choose best strategy between provided ones

        Parameters
        ----------
        labels : list
            Strategy labels to compare.
        parallel : bool, optional
            Perform `n_best` strategies evaluation in parallel.
            True by default.
        use_cache : bool, optional
            Use solver cache if available. True by default.

        Returns
        -------
        dict
            Results as a dictionary.
        """
        n_best = self._learner.options['n_best']

        # For each n_best classes get x, y, time and store the best one
        x = []
        time = []
        infeas = []
        cost = []

        strategies = [self.encoding[label] for label in labels]

        # Cache is a list of solver caches to pass
        cache = [None] * n_best
        if self._solver_cache and use_cache:
            cache = [self._solver_cache[label] for label in labels]

        n_jobs = u.get_n_processes(n_best) if parallel else 1

        results = Parallel(n_jobs=n_jobs)(
            delayed(self._problem.solve)(strategy=strategies[j],
                                         cache=cache[j])
            for j in range(n_best))

        x = [r["x"] for r in results]
        time = [r["time"] for r in results]
        infeas = [r["infeasibility"] for r in results]
        cost = [r["cost"] for r in results]

        # Pick best class between k ones
        infeas = np.array(infeas)
        cost = np.array(cost)
        idx_filter = np.where(infeas <= stg.INFEAS_TOL)[0]
        if len(idx_filter) > 0:
            # Case 1: Feasible points
            # -> Get solution with best cost
            #    between feasible ones
            if self._problem.sense() == Minimize:
                idx_pick = idx_filter[np.argmin(cost[idx_filter])]
            elif self._problem.sense() == Maximize:
                idx_pick = idx_filter[np.argmax(cost[idx_filter])]
            else:
                e.value_error('Objective type not understood')
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
              message="Predict optimal solution",
              use_cache=True,
              verbose=False,
              ):
        """
        Predict optimal solution given the parameters X.

        Parameters
        ----------
        X : pandas DataFrame or Series
            Data points.
        use_cache : bool, optional
            Use solver cache?  Defaults to True.

        Returns
        -------
        list
            List of result dictionaries.
        """

        if isinstance(X, pd.Series):
            X = pd.DataFrame(X).transpose()
        n_points = len(X)

        if use_cache and not self._solver_cache:
            e.warning("Solver cache requested but the cache has "
                      "not been computed for this problem. "
                      "Possibly parameters in proble matrices.")

        # Change verbose setting
        if verbose:
            self._problem.verbose = True

        # Define array of results to return
        results = []

        # Predict best n_best classes for all the points
        X_pred = pandas2array(X)
        t_start = time()
        classes = self._learner.predict(X_pred)
        t_predict = (time() - t_start) / n_points  # Average predict time

        if n_points > 1:
            stg.logger.info(message)
            ran = tqdm(range(n_points))
        else:
            # Do not print anything if just one point
            ran = range(n_points)

        for i in ran:

            # Populate problem with i-th data point
            self._problem.populate(X.iloc[i])

            results.append(self.choose_best(classes[i, :],
                                            use_cache=use_cache))

        # Append predict time
        for r in results:
            r['pred_time'] = t_predict
            r['solve_time'] = r['time']
            r['time'] = r['pred_time'] + r['solve_time']

        if len(results) == 1:
            results = results[0]

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
            e.value_error("You cannot save the optimizer without " +
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
                file_dict = {'_problem': self._problem,
                             # '_solver_cache': self._solver_cache,  # Cannot pickle
                             'learner_name': self._learner.name,
                             'learner_options': self._learner.options,
                             'encoding': self.encoding
                             }
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
            e.value_error("File %s does not exist." % file_name)

        # Extract file to temporary directory and read it
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(file_name) as tar:
                tar.extractall(path=tmpdir)

            # Load optimizer
            optimizer_file_name = os.path.join(tmpdir, "optimizer.pkl")
            if not optimizer_file_name:
                e.value_error("Optimizer pkl file does not exist.")
            with open(optimizer_file_name, "rb") as f:
                optimizer_dict = pkl.load(f)

            name = optimizer_dict.get('name', 'problem')

            # Create optimizer using loaded dict
            problem = optimizer_dict['_problem'].cvxpy_problem
            optimizer = cls(problem, name=name)

            # Assign strategies encoding
            optimizer.encoding = optimizer_dict['encoding']
            optimizer._sampler = optimizer_dict.get('_sampler', None)

            # Load learner
            learner_name = optimizer_dict['learner_name']
            learner_options = optimizer_dict['learner_options']
            optimizer._learner = \
                LEARNER_MAP[learner_name](n_input=optimizer.n_parameters,
                                          n_classes=len(optimizer.encoding),
                                          **learner_options)
            optimizer._learner.load(os.path.join(tmpdir, "learner"))

        return optimizer

    def performance(self, theta,
                    parallel=False,
                    use_cache=True):
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

        stg.logger.info("Performance evaluation")
        # Get strategy for each point
        results_test = self._problem.solve_parametric(theta,
                                                      parallel=parallel,
                                                      message="Compute " +
                                                      "tight constraints " +
                                                      "for test set")

        time_test = [r['time'] for r in results_test]
        cost_test = [r['cost'] for r in results_test]

        # Get predicted strategy for each point
        results_pred = self.solve(theta,
                                  message="Predict tight constraints for " +
                                  "test set",
                                  use_cache=use_cache)
        time_pred = [r['time'] for r in results_pred]
        solve_time_pred = [r['solve_time'] for r in results_pred]
        pred_time_pred = [r['pred_time'] for r in results_pred]
        cost_pred = [r['cost'] for r in results_pred]
        infeas = np.array([r['infeasibility'] for r in results_pred])

        n_test = len(theta)
        n_train = self._learner.n_train  # Number of training samples
        n_theta = n_features(theta)  # Number of parameters
        n_strategies = len(self.encoding)  # Number of strategies

        # Compute comparative statistics
        time_comp = np.array([time_test[i] / time_pred[i]
                              for i in range(n_test)])
        subopt = np.array([suboptimality(cost_pred[i], cost_test[i],
                                         self._problem.sense())
                           for i in range(n_test)])

        # accuracy
        test_accuracy, idx_correct = accuracy(results_pred, results_test,
                                              self._problem.sense())

        # Time statistics
        avg_time_improv = np.mean(time_test) / np.mean(time_pred)
        max_time_improv = np.max(time_test) / np.max(time_pred)

        # Create dataframes to return
        df = pd.Series(
            {
                "problem": self.name,
                "learner": self._learner.name,
                "n_best": self._learner.options['n_best'],
                "n_var": self._problem.n_var,
                "n_constr": self._problem.n_constraints,
                "n_test": n_test,
                "n_train": n_train,
                "n_theta": n_theta,
                "good_turing": self._sampler.good_turing,
                "good_turing_smooth": self._sampler.good_turing_smooth,
                "n_correct": np.sum(idx_correct),
                "n_strategies": n_strategies,
                "accuracy": 100 * test_accuracy,
                "n_infeas": np.sum(infeas >= stg.INFEAS_TOL),
                "avg_infeas": np.mean(infeas),
                "std_infeas": np.std(infeas),
                "avg_subopt": np.mean(subopt[np.where(infeas <=
                                      stg.INFEAS_TOL)[0]]),
                "std_subopt": np.std(subopt[np.where(infeas <=
                                     stg.INFEAS_TOL)[0]]),
                "max_infeas": np.max(infeas),
                "max_subopt": np.max(subopt),
                "mean_solve_time_pred": np.mean(solve_time_pred),
                "std_solve_time_pred": np.std(solve_time_pred),
                "mean_pred_time_pred": np.mean(pred_time_pred),
                "std_pred_time_pred": np.std(pred_time_pred),
                "mean_time_pred": np.mean(time_pred),
                "std_time_pred": np.std(time_pred),
                "mean_time_full": np.mean(time_test),
                "std_time_full": np.std(time_test),
                "avg_time_improv": avg_time_improv,
                "max_time_improv": max_time_improv,
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
                "solve_time_pred": solve_time_pred,
                "pred_time_pred": pred_time_pred,
                "time_pred": time_pred,
                "time_full": time_test,
                "time_improvement": time_comp,
            }
        )

        return df, df_detail
