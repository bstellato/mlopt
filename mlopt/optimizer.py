from mlopt.problem import Problem  # , _solve_with_strategy_multiprocess
from multiprocessing import Pool
from datetime import datetime
#  from pathos.multiprocessing import ProcessPool as Pool
from itertools import repeat
import cvxpy as cp
from mlopt.settings import DEFAULT_SOLVER, DEFAULT_LEARNER, INFEAS_TOL, \
    ALPHA_CONDENSE, K_MAX_STRATEGIES, DIVISION_TOL, PREFILTER_STRATEGY_NUM  # , CONDENSE_CHECK
from mlopt.learners import LEARNER_MAP
from mlopt.sampling import Sampler
from mlopt.strategy import encode_strategies, strategy_distance
from mlopt.utils import n_features, accuracy, suboptimality
from mlopt.utils import get_n_processes
from mlopt.kkt import KKT, create_kkt_matrix
#  from pypardiso import factorized
from time import time
from scipy.sparse.linalg import factorized
import cvxpy.settings as cps
import pandas as pd
import numpy as np
import os
from glob import glob
import tempfile
import tarfile
import pickle as pkl
from tqdm import tqdm
import logging
import ray


@ray.remote
def _prefilter_strategies_ray(y_i, encoding):
    """Ray wrapper."""
    return _prefilter_strategies(y_i, encoding)


def _prefilter_strategies(y_i, encoding):
    s_i = encoding[y_i]
    dist_i = np.array([strategy_distance(s_i, s) for s in encoding])

    # Sort distances and pick PREFILTER_STRATEGY_NUM ones
    idx_dist_i = np.argsort(dist_i)[:PREFILTER_STRATEGY_NUM]

    return idx_dist_i.tolist()


@ray.remote
def _compute_cost_differences_ray(theta, obj_train,
                                  alpha_strategies_filtered,
                                  problem, encoding):
    """Ray wrapper."""
    return _compute_cost_differences(theta, obj_train,
                                     alpha_strategies_filtered,
                                     problem, encoding)


def _compute_cost_differences(theta, obj_train,
                              alpha_strategies_filtered,
                              problem, encoding):
    """
    Compute cost differences for sample theta.

    To be used in parallel multiprocessing.

    """

    n_strategies = len(encoding)

    problem.populate(theta)  # Populate parameters

    c = {}
    alpha_strategies = []

    # Serialized solution over the strategies
    results = {j: problem.solve_with_strategy(encoding[j])
               for j in alpha_strategies_filtered}

    # Process results
    for j in alpha_strategies_filtered:
        diff = np.abs(results[j]['cost'] - obj_train)
        if np.abs(obj_train) > DIVISION_TOL:  # Normalize in case
            diff /= np.abs(obj_train)

        if diff < ALPHA_CONDENSE and \
                results[j]['infeasibility'] < INFEAS_TOL:
            alpha_strategies.append(j)
            c[j] = diff

    # Check for consistency. At least one strategy working per point.
    if len(alpha_strategies) == 0:
        # DEBUG: Dump file to check
        import pickle
        temp_file = 'log_' + \
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pkl"
        temp_dict = {'problem': problem,
                     'encoding': encoding,
                     'theta': theta,
                     'obj_train': obj_train,
                     'alpha_strategies': alpha_strategies,
                     'alpha_strategies_filtered': alpha_strategies_filtered,
                     'c': c,
                     'results': results,
                     'ALPHA_CONDENSE': ALPHA_CONDENSE,
                     'INFEAS_TOL': INFEAS_TOL,
                     'DIVISION_TOL': DIVISION_TOL}
        with open(temp_file, 'wb') as handle:
            pickle.dump(temp_dict, handle)

    logging.debug("Kept %d/%d points" % (len(alpha_strategies), n_strategies))

    return alpha_strategies, c


class Optimizer(object):
    """
    Machine Learning Optimizer class.
    """

    def __init__(self,
                 objective, constraints,
                 name="problem",
                 log_level=logging.WARNING,
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

        logging.basicConfig(level=log_level)

        self._problem = Problem(objective, constraints,
                                solver=DEFAULT_SOLVER,
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
            err = "Model has been trained yet to " + \
                "return the number of strategies."
            logging.error(err)
            raise ValueError(err)

        return len(self.encoding)

    def variables(self):
        """Problem variables."""
        return self._problem.variables()

    def parameters(self):
        """Problem parameters."""
        return self._problem.parameters()

    @property
    def n_parameters(self):
        """Number of parameters."""
        return self._problem.n_parameters

    def samples_present(self):
        """Check if samples have been generated."""
        return (self.X_train is not None) and \
            (self.y_train is not None) and \
            (self.encoding is not None)

    def sample(self, sampling_fn, parallel=True):
        """
        Sample parameters.
        """

        # Create sampler
        self._sampler = Sampler(self._problem, sampling_fn)

        # Sample parameters
        self.X_train, self.y_train, self.encoding = \
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
            err = "You need to get the strategies " + \
                "from the data first by training the model."
            logging.error(err)
            raise ValueError(err)

        # Save to file
        with open(file_name, 'wb') \
                as data:
            data_dict = {'X_train': self.X_train,
                         'y_train': self.y_train,
                         'obj_train': self.obj_train,
                         '_problem': self._problem,
                         'encoding': self.encoding}

            # Store full encodings backup if condensing happened
            if hasattr(self, 'y_train_full') and \
                    hasattr(self, 'encoding_full'):
                data_dict['y_train_full'] = self.y_train_full
                data_dict['encoding_full'] = self.encoding_full

            if hasattr(self, '_c') and \
                    hasattr(self, '_alpha_strategies'):
                data_dict['_c'] = self._c
                data_dict['_alpha_strategies'] = self._alpha_strategies

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
            raise ValueError("File %s does not exist." % file_name)

        # Load optimizer
        with open(file_name, "rb") as f:
            data_dict = pkl.load(f)

        # Store data internally
        self.X_train = data_dict['X_train']
        self.y_train = data_dict['y_train']
        self.obj_train = data_dict['obj_train']
        self._problem = data_dict['_problem']
        self.encoding = data_dict['encoding']

        # Full strategies backup after condensing
        if ('y_train_full' in data_dict) and \
                ('encoding_full' in data_dict):
            self.y_train_full = data_dict['y_train_full']
            self.encoding_full = data_dict['encoding_full']

        if ('_c' in data_dict) and \
                ('_alpha_strategies' in data_dict):
            self._c = data_dict['_c']
            self._alpha_strategies = data_dict['_alpha_strategies']

        # Compute Good turing estimates
        self._sampler = Sampler(self._problem, n_samples=len(self.X_train))
        self._sampler.compute_good_turing(self.y_train)

    def _get_samples(self, X=None, sampling_fn=None, parallel=True,
                     condense_strategies=True):
        """Get samples either from data or from sampling function"""
        # Assert we have data to train or already trained
        if X is None and sampling_fn is None and not self.samples_present():
            err = "Not enough arguments to train the model"
            logging.error(err)
            raise ValueError(err)

        if X is not None and sampling_fn is not None:
            err = "You can pass only one value between X and sampling_fn"
            logging.error(err)
            raise ValueError(err)

        # Check if data is passed, otherwise train
        #  if (X is not None) and not self.samples_present():
        if X is not None:
            logging.info("Use new data")
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
                                   if 'strategy' not in x.keys()}
            if not_feasible_points:
                e = "number of infeasible points %d" % len(not_feasible_points)
                logging.error(e)
                raise ValueError(e)

            self.obj_train = [r['cost'] for r in results]
            train_strategies = [r['strategy'] for r in results]

            # Check if the problems are solvable
            for r in results:
                assert r['status'] in cps.SOLUTION_PRESENT, \
                    "The training points must be feasible"

            # Encode strategies
            self.y_train, self.encoding = \
                encode_strategies(train_strategies)

            # Compute Good turing estimates
            self._sampler = Sampler(self._problem, n_samples=len(self.X_train))
            self._sampler.compute_good_turing(self.y_train)

        elif sampling_fn is not None and not self.samples_present():
            logging.info("Use iterative sampling")
            # Create X_train, y_train and encoding from
            # sampling function
            self.sample(sampling_fn, parallel=parallel)

        # Condense strategies
        if (len(self.encoding) > K_MAX_STRATEGIES) and \
                condense_strategies:
            self.condense_strategies(parallel=parallel)

    #  def _prefilter_strategies(self):
    #      """Prefilter strategies for single sample i"""
    #      n_samples = len(self.X_train)
    #      n_strategies = len(self.encoding)
    #      alpha_strategies = [[] for _ in range(n_samples)]
    #
    #      logging.info("Prefiltering strategies up %d per sample" %
    #                   PREFILTER_STRATEGY_NUM)
    #      n_filter = 0
    #      for i in tqdm(range(n_samples)):
    #          s_i = self.encoding[self.y_train[i]]
    #          dist_i = np.array([strategy_distance(s_i, s)
    #                             for s in self.encoding])
    #          n_check = 0
    #
    #          # Sort distances and pick PREFILTER_STRATEGY_NUM ones
    #          idx_dist_i = np.argsort(dist_i)[:PREFILTER_STRATEGY_NUM]
    #
    #          # Pick distances
    #          for j in idx_dist_i:
    #              alpha_strategies[i].append(j)
    #              n_filter += 1
    #              n_check += 1
    #
    #          #  for j in range(n_strategies):
    #          #      if dist_i[j] < PREFILTER_STRATEGY_DIST:
    #          #          alpha_strategies[i].append(j)
    #          #          n_filter += 1
    #          #          n_check += 1
    #
    #          if n_check == 0:
    #              e = "No strategy kept for point %i" % i
    #              logging.error(e)
    #              raise ValueError(e)
    #
    #      logging.info("Filtered %d/%d points" % (n_filter, n_samples * n_strategies))
    #
    #      return alpha_strategies

    def _compute_sample_strategy_pairs(self, solver=DEFAULT_SOLVER,
                                       k_max_strategies=K_MAX_STRATEGIES,
                                       parallel=True):

        n_samples = len(self.X_train)
        n_strategies = len(self.encoding)

        logging.info("Computing sample-strategy pairs")
        logging.info("n_samples = %d, n_strategies = %d" %
                     (n_samples, n_strategies))

        # Compute costs
        c = {}

        if parallel:
            theta_arr = [self.X_train.iloc[i] for i in range(n_samples)]
            n_proc = get_n_processes(n_samples)

            tmp_dir = './ray_tmp/' + \
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "/"
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
            ray.init(num_cpus=n_proc,
                     redis_max_memory=(1024**2)*500,  # .1GB  # CAP
                     #  object_store_memory=(1024**2)*1000,  # .1GB
                     temp_dir=tmp_dir
                     )

            # Problem is not serialized correctly
            ray.register_custom_serializer(Problem, use_pickle=True)

            # Share encoding between all processors
            encoding_id = ray.put(self.encoding)

            # Pre-filter strategies with max strategies per point
            alpha_strategies = [[] for _ in range(n_samples)]

            logging.info("Prefiltering strategies up to %d per sample " %
                         PREFILTER_STRATEGY_NUM +
                         "(parallel %i processors)" %
                         n_proc)
            n_filter = 0
            result_ids = []
            for i in range(n_samples):
                result_ids.append(
                    _prefilter_strategies_ray.remote(self.y_train[i],
                                                     encoding_id))

            for i in tqdm(range(n_samples)):
                alpha_strategies[i] = ray.get(result_ids[i])

                if len(alpha_strategies[i]) == 0:
                    e = "No strategy kept for point %i" % i
                    logging.error(e)
                    raise ValueError(e)

                n_filter += len(alpha_strategies[i])

            logging.info("Filtered %d/%d points" % (n_filter,
                                                    n_samples * n_strategies))

            # Condense strategies
            logging.info("Computing sample_strategy pairs "
                         "(parallel %i processors)..." %
                         n_proc)
            result_ids = []
            for i in range(n_samples):
                result_ids.append(
                    _compute_cost_differences_ray.remote(theta_arr[i],
                                                         self.obj_train[i],
                                                         alpha_strategies[i],
                                                         self._problem,
                                                         encoding_id))

            for i in tqdm(range(n_samples)):
                alpha_strategies[i], c_i = ray.get(result_ids[i])

                if len(alpha_strategies[i]) == 0:
                    e = "No good strategy for point %d" % i
                    logging.error(e)
                    raise ValueError(e)

                c.update({(i, j): val for j, val in c_i.items()})

            ray.shutdown()

            # Old solution with multiprocessing
            #  pool = Pool(processes=n_proc)
            #  logging.info("Computing alpha strategies "
            #               "(parallel %i processors)..." %
            #               n_proc)
            #
            #  #  pbar = tqdm(total=n_samples)
            #  #  results = []
            #  #  for i in range(n_samples):
            #  #      pool.apply_async(_compute_cost_differences,
            #  #                       (i, theta_arr[i], self.obj_train[i],
            #  #                        self._problem, self.encoding),
            #  #                       callback=append_async_results)
            #
            #  #  results = list(tqdm(pool.imap_unordered(
            #  #      _compute_cost_differences,
            #  #      zip(range(n_samples), theta_arr, self.obj_train,
            #  #          repeat(self._problem), repeat(self.encoding))
            #  #      ), total=n_samples))
            #
            #  results = pool.imap(
            #      self._compute_cost_differences,
            #      zip(range(n_samples), theta_arr, self.obj_train,
            #          repeat(self._problem), repeat(self.encoding))
            #      )
            #
            #  for r in tqdm(results, total=n_samples):
            #      i, c_i = r[-1], r[1]
            #      alpha_strategies[i] = r[0]
            #      c.update({(i, j): val for j, val in c_i.items()})
            #
            #  pool.close()
            #  pool.join()
            #  pbar.close()

        else:

            # Pre-filter strategies with max strategies per point
            alpha_strategies = [[] for _ in range(n_samples)]

            logging.info("Prefiltering strategies up to %d per sample (serial)" %
                         PREFILTER_STRATEGY_NUM)
            n_filter = 0
            for i in tqdm(range(n_samples)):
                alpha_strategies[i] = _prefilter_strategies(self.y_train[i],
                                                            self.encoding)
                if len(alpha_strategies[i]) == 0:
                    e = "No strategy kept for point %i" % i
                    logging.error(e)
                    raise ValueError(e)

                n_filter += len(alpha_strategies[i])

            logging.info("Filtered %d/%d points" % (n_filter,
                                                    n_samples * n_strategies))

            logging.info("Computing alpha strategies (serial)")
            for i in tqdm(range(n_samples)):
                alpha_strategies[i], c_i = \
                    _compute_cost_differences(self.X_train.iloc[i],
                                              self.obj_train[i],
                                              alpha_strategies[i],
                                              self._problem,
                                              self.encoding)
                c.update({(i, j): val for j, val in c_i.items()})

        n_pairs = sum(len(x) for x in alpha_strategies)
        logging.info("Pruned %.2f %% suboptimal pairs" %
                     (100*(1 - n_pairs / (n_samples * n_strategies))))
        logging.info("Remaining number of pairs %d" % n_pairs)

        self._c = c
        self._alpha_strategies = alpha_strategies

    def condense_strategies(self,
                            k_max_strategies=K_MAX_STRATEGIES,
                            parallel=True):
        """Condense strategies using MIO.


        Parameters
        ----------
        solver : string
            Solver to use.
        k_max_strategies : int
            Maximum number of strategies allowed.
        parallel : bool
            Parallelize strategies computations over samples.
        """
        if not hasattr(self, '_c') or \
                not hasattr(self, '_alpha_strategies'):
            self._compute_sample_strategy_pairs(parallel=parallel)
        c = self._c
        alpha_strategies = self._alpha_strategies

        n_samples = len(self.X_train)
        n_strategies = len(self.encoding)

        logging.info("Formulating and solving MIO condensing problem.")

        # Get alpha_samples
        alpha_samples = [[] for _ in range(n_strategies)]
        M = np.zeros(n_strategies)
        # Search over all alpha_strategies
        for i in range(n_samples):
            for j in alpha_strategies[i]:
                alpha_samples[j].append(i)
                M[j] += 1

        # Formulate with Gurobi directly
        import gurobipy as grb
        model = grb.Model()

        # Variables
        x = {(i, j): model.addVar(vtype=grb.GRB.BINARY,
                                  name='x_%d,%d' % (i, j))
             for i in range(n_samples)
             for j in alpha_strategies[i]}
        y = {j: model.addVar(vtype=grb.GRB.BINARY,
                             name='y_%d' % j)
             for j in range(n_strategies)}

        # Constraints
        for i in range(n_samples):
            model.addConstr(grb.quicksum(x[i, j]
                                         for j in alpha_strategies[i]) == 1)
        for i in range(n_samples):
            for j in alpha_strategies[i]:
                model.addConstr(x[i, j] <= y[j])
        model.addConstr(grb.quicksum(y[j] for j in range(n_strategies))
                        <= k_max_strategies)

        for j in range(n_strategies):
            model.addConstr(y[j] <=
                            M[j] * grb.quicksum(x[i, j]
                                                for i in alpha_samples[j]))

        # Objective
        model.setObjective(grb.quicksum(c[i, j] * x[i, j]
                                        for i in range(n_samples)
                                        for j in alpha_strategies[i]))

        # Solve
        model.setParam("OutputFlag", 0)
        model.optimize()
        assert model.Status == grb.GRB.OPTIMAL  # Assert optimal solution

        # Get solution
        x_opt = {(i, j): x[i, j].X
                 for i in range(n_samples)
                 for j in alpha_strategies[i]}
        y_opt = np.array([y[j].X for j in range(n_strategies)])
        degradation = 1. / n_samples * np.sum(c[i, j] * x_opt[i, j]
                                              for i in range(n_samples)
                                              for j in alpha_strategies[i])

        logging.info("Average cost degradation = %.2e %%" %
                     (100 * degradation))

        # Get chosen strategies
        chosen_strategies = np.where(y_opt == 1)[0]

        logging.info("Number of chosen strategies %d" % len(chosen_strategies))

        # Backup full strategies
        self.encoding_full = self.encoding
        self.y_train_full = self.y_train

        # Assign new labels and encodings
        self.encoding = [self.encoding[i] for i in chosen_strategies]
        self.y_train = -1 * np.ones(n_samples, dtype=int)

        for i in range(n_samples):
            # Get best strategy per sample
            for j in alpha_strategies[i]:
                if x_opt[i, j] == 1:
                    self.y_train[i] = np.where(chosen_strategies == j)[0][0]
                    break
            if self.y_train[i] == -1:
                raise ValueError("No strategy selected for sample %d" % i)

    def train(self, X=None, sampling_fn=None,
              parallel=True,
              learner=DEFAULT_LEARNER,
              condense_strategies=True,
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
        self._get_samples(X, sampling_fn, parallel,
                          condense_strategies=condense_strategies)

        # Define learner
        self._learner = LEARNER_MAP[learner](n_input=n_features(self.X_train),
                                             n_classes=len(self.encoding),
                                             **learner_options)

        # Train learner
        self._learner.train(self.X_train, self.y_train)

        # Add factorization faching if
        # 1. Problem is MIQP
        # TODO: Add the second point!
        # 2. Parameters enter only in the problem vectors
        if self._problem.is_qp():
            logging.info("Caching KKT solver factors for each strategy "
                         "(it works only for QP-representable problems "
                         "with parameters only in constraints RHS)")
            self._cache_factors()

    def _cache_factors(self):
        """Cache linear system solver factorizations"""

        self._solver_cache = []
        for strategy_idx in range(self.n_strategies):

            # Get a parameter giving that strategy
            strategy = self.encoding[strategy_idx]
            idx_param = np.where(self.y_train == strategy_idx)[0]
            theta = self.X_train.iloc[idx_param[0]]

            self._problem.populate(theta)

            self._problem._relax_disc_var()

            reduced_problem = \
                self._problem._construct_reduced_problem(strategy)

            data, full_chain, inv_data = \
                reduced_problem.get_problem_data(solver=KKT)

            KKT_mat = create_kkt_matrix(data)
            solve_kkt = factorized(KKT_mat)

            cache = {}
            cache['factors'] = solve_kkt
            cache['inverse_data'] = inv_data
            cache['chain'] = full_chain

            self._solver_cache += [cache]

            self._problem._restore_disc_var()

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
            Use solver cache? True by default.

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

        strategies = [self.encoding[l] for l in labels]

        # Cache is a list of solver caches to pass
        cache = [None] * n_best
        if self._solver_cache and use_cache:
            cache = [self._solver_cache[l] for l in labels]

        if parallel:
            n_proc = get_n_processes(max_n=n_best)
            pool = Pool(processes=n_proc)
            results = pool.starmap(self._problem.solve_with_strategy,
                                   zip(strategies, cache))
            x = [r["x"] for r in results]
            time = [r["time"] for r in results]
            infeas = [r["infeasibility"] for r in results]
            cost = [r["cost"] for r in results]
        else:
            for j in range(n_best):
                res = self._problem.solve_with_strategy(strategies[j],
                                                        cache[j])
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
              message="Predict optimal solution",
              parallel=False,
              use_cache=True,
              verbose=False,
              ):
        """
        Predict optimal solution given the parameters X.

        Parameters
        ----------
        X : pandas DataFrame or Series
            Data points.
        parallel : bool, optional
            Perform `n_best` strategies evaluation in parallel.
            Defaults to True.
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
        n_best = self._learner.options['n_best']

        if use_cache and not self._solver_cache:
            err = "Solver cache requested but the cache has not been" + \
                "computed for this problem. Is it MIQP representable?"
            logging.error(err)
            raise ValueError(err)

        # Change verbose setting
        if verbose:
            self._problem.solver_options['verbose'] = True

        # Define array of results to return
        results = []

        # Predict best n_best classes for all the points
        t_start = time()
        classes = self._learner.predict(X)
        t_predict = (time() - t_start) / n_points  # Compute average predict time

        if parallel:
            n_proc = get_n_processes(max_n=n_best)
            message = message + "in parallel (%d processors)" % n_proc
        else:
            message = message + " in serial"

        for i in tqdm(range(n_points), desc=message):

            # Populate problem with i-th data point
            self._problem.populate(X.iloc[i])

            # Pick strategies from encoding
            #  strategies = [self.encoding[classes[i, j]]
            #                for j in range(n_best)]
            #  results.append(self.choose_best(strategies,
            #                                  parallel=parallel))

            results.append(self.choose_best(classes[i, :],
                                            parallel=parallel,
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
                file_dict = {'_problem': self._problem,
                             '_solver_cache': self._solver_cache,
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
            problem = optimizer_dict['_problem'].cvxpy_problem
            optimizer = cls(problem.objective,
                            problem.constraints,
                            name=optimizer_dict['name'])

            # Assign strategies encoding
            optimizer.encoding = optimizer_dict['encoding']
            optimizer._sampler = optimizer_dict['_sampler']

            # Load learner
            learner_name = optimizer_dict['learner_name']
            learner_options = optimizer_dict['learner_options']
            optimizer._learner = \
                LEARNER_MAP[learner_name](n_input=optimizer.n_parameters,
                                          n_classes=len(optimizer.encoding),
                                          **learner_options)
            optimizer._learner.load(os.path.join(tmpdir, "learner"))

        # Compute Good turing estimates
        optimizer._sampler = Sampler(optimizer._problem,
                                     n_samples=len(optimizer.X_train))
        optimizer._sampler.compute_good_turing(optimizer.y_train)

        return optimizer

    def performance(self, theta,
                    parallel=True,
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

        logging.info("Performance evaluation")
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
        subopt = np.array([suboptimality(cost_pred[i], cost_test[i])
                           for i in range(n_test)])

        # accuracy
        test_accuracy, idx_correct = accuracy(results_pred, results_test)

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
                "n_infeas": np.sum(infeas >= INFEAS_TOL),
                "avg_infeas": np.mean(infeas),
                "std_infeas": np.std(infeas),
                "avg_subopt": np.mean(subopt[np.where(infeas <=
                                      INFEAS_TOL)[0]]),
                "std_subopt": np.std(subopt[np.where(infeas <=
                                     INFEAS_TOL)[0]]),
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
