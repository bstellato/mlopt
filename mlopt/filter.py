import ray
import numpy as np
import logging
from mlopt.problem import solve_with_strategy
from mlopt.strategy import strategy_distance
import mlopt.settings as stg
from mlopt.problem import Problem
from tqdm import tqdm
import os


@ray.remote
def prefilter_strategies_ray(y_i, encoding):
    """Ray wrapper."""
    return prefilter_strategies(y_i, encoding)


def prefilter_strategies(y_i, encoding):
    s_i = encoding[y_i]
    dist_i = np.array([strategy_distance(s_i, s) for s in encoding])

    # Sort distances and pick PREFILTER_STRATEGY_NUM ones
    idx_dist_i = np.argsort(dist_i)[:stg.PREFILTER_STRATEGY_NUM]

    return idx_dist_i.tolist()


@ray.remote
def compute_cost_differences_ray(theta, obj_train,
                                 prefiltered_strategies,
                                 problem, encoding):
    """Ray wrapper."""
    return compute_cost_differences(theta, obj_train,
                                    prefiltered_strategies,
                                    problem, encoding)


def compute_cost_differences(theta, obj_train,
                             prefiltered_strategies,
                             problem, encoding):
    """
    Compute cost differences for sample theta.

    To be used in parallel multiprocessing.

    """

    n_strategies = len(encoding)

    problem.populate(theta)  # Populate parameters

    c = {}
    filtered_strategies = []

    # Serialized solution over the strategies
    results = {j: solve_with_strategy(problem, encoding[j])
               for j in prefiltered_strategies}

    # Process results
    for j in prefiltered_strategies:
        diff = np.abs(results[j]['cost'] - obj_train)
        if np.abs(obj_train) > stg.DIVISION_TOL:  # Normalize in case
            diff /= np.abs(obj_train)

        if diff < stg.FILTER_SUBOPT and \
                results[j]['infeasibility'] < stg.INFEAS_TOL:
            filtered_strategies.append(j)
            c[j] = diff

    # Check for consistency. At least one strategy working per point.
    if len(filtered_strategies) == 0:
        # DEBUG: Dump file to check
        import pickle
        from datetime import datetime as dt
        temp_file = 'log_' + \
            dt.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pkl"
        temp_dict = {'problem': problem,
                     'encoding': encoding,
                     'theta': theta,
                     'obj_train': obj_train,
                     'filtered_strategies': filtered_strategies,
                     'prefiltered_strategies': prefiltered_strategies,
                     'c': c,
                     'results': results,
                     'FILTER_SUBOPT': stg.FILTER_SUBOPT,
                     'INFEAS_TOL': stg.INFEAS_TOL,
                     'DIVISION_TOL': stg.DIVISION_TOL}
        with open(temp_file, 'wb') as handle:
            pickle.dump(temp_dict, handle)

    logging.debug("Kept %d/%d points" %
                  (len(filtered_strategies), n_strategies))

    return filtered_strategies, c


class Filter(object):
    """Strategy filter."""
    def __init__(self,
                 X_train=None,
                 y_train=None,
                 obj_train=None,
                 encoding=None,
                 problem=None):
        """Initialize strategy condenser."""
        self.X_train = X_train
        self.y_train = y_train
        self.encoding = encoding
        self.obj_train = obj_train
        self.problem = problem

    def compute_sample_strategy_pairs(self, solver=stg.DEFAULT_SOLVER,
                                      k_max_strategies=stg.K_MAX_STRATEGIES,
                                      parallel=True):

        n_samples = len(self.X_train)
        n_strategies = len(self.encoding)

        logging.info("Computing sample-strategy pairs")
        logging.info("n_samples = %d, n_strategies = %d" %
                     (n_samples, n_strategies))

        # Compute costs
        c = {}

        if parallel:

            # Share encoding between all processors
            encoding_id = ray.put(self.encoding)

            # Pre-filter strategies with max strategies per point
            filtered_strategies = [[] for _ in range(n_samples)]

            logging.info("Prefiltering strategies up to %d per sample "
                         % stg.PREFILTER_STRATEGY_NUM +
                         "(parallel)")
            n_filter = 0
            result_ids = []
            for i in range(n_samples):
                result_ids.append(
                    prefilter_strategies_ray.remote(self.y_train[i],
                                                    encoding_id))

            for i in tqdm(range(n_samples)):
                filtered_strategies[i] = ray.get(result_ids[i])

                if len(filtered_strategies[i]) == 0:
                    e = "No strategy kept for point %i" % i
                    logging.error(e)
                    raise ValueError(e)

                n_filter += len(filtered_strategies[i])

            logging.info("Filtered %d/%d points" % (n_filter,
                                                    n_samples * n_strategies))

            # Condense strategies
            logging.info("Computing sample_strategy pairs (parallel)")
            result_ids = []
            for i in range(n_samples):
                result_ids.append(
                    compute_cost_differences_ray.remote(self.X_train.iloc[i],
                                                        self.obj_train[i],
                                                        filtered_strategies[i],
                                                        self.problem,
                                                        encoding_id))

            for i in tqdm(range(n_samples)):
                filtered_strategies[i], c_i = ray.get(result_ids[i])

                if len(filtered_strategies[i]) == 0:
                    e = "No good strategy for point %d" % i
                    logging.error(e)
                    raise ValueError(e)

                c.update({(i, j): val for j, val in c_i.items()})

            ray.shutdown()

        else:

            # Pre-filter strategies with max strategies per point
            filtered_strategies = [[] for _ in range(n_samples)]

            logging.info("Prefiltering strategies up to %d" %
                         stg.PREFILTER_STRATEGY_NUM +
                         " per sample (serial)")
            n_filter = 0
            for i in tqdm(range(n_samples)):
                filtered_strategies[i] = prefilter_strategies(self.y_train[i],
                                                              self.encoding)
                if len(filtered_strategies[i]) == 0:
                    e = "No strategy kept for point %i" % i
                    logging.error(e)
                    raise ValueError(e)

                n_filter += len(filtered_strategies[i])

            logging.info("Filtered %d/%d points" % (n_filter,
                                                    n_samples * n_strategies))

            logging.info("Computing filtered strategies (serial)")
            for i in tqdm(range(n_samples)):
                filtered_strategies[i], c_i = \
                    compute_cost_differences(self.X_train.iloc[i],
                                             self.obj_train[i],
                                             filtered_strategies[i],
                                             self.problem,
                                             self.encoding)
                c.update({(i, j): val for j, val in c_i.items()})

        n_pairs = sum(len(x) for x in filtered_strategies)
        logging.info("Pruned %.2f %% suboptimal pairs" %
                     (100*(1 - n_pairs / (n_samples * n_strategies))))
        logging.info("Remaining number of pairs %d" % n_pairs)

        self.c = c  # Cost distances
        self.filtered_strategies = filtered_strategies

    def filter(self,
               k_max_strategies=stg.K_MAX_STRATEGIES,
               parallel=True):
        """Filter strategies using MIO.


        Parameters
        ----------
        solver : string
            Solver to use.
        k_max_strategies : int
            Maximum number of strategies allowed.
        parallel : bool
            Parallelize strategies computations over samples.
        """
        if not hasattr(self, 'c') or \
                not hasattr(self, 'filtered_strategies'):
            self.compute_sample_strategy_pairs(parallel=parallel)
        c, filtered_strategies = self.c, self.filtered_strategies

        n_samples = len(self.X_train)
        n_strategies = len(self.encoding)

        logging.info("Formulating and solving MIO condensing problem.")
        logging.info("Maximum %d strategies" % k_max_strategies)

        # Get samples assigned to filtered strategies
        filtered_samples = [[] for _ in range(n_strategies)]
        M = np.zeros(n_strategies)
        # Search over all filtered_strategies
        for i in range(n_samples):
            for j in filtered_strategies[i]:
                filtered_samples[j].append(i)
                M[j] += 1

        # Formulate with Gurobi directly
        import gurobipy as grb
        model = grb.Model()

        # Variables
        x = {(i, j): model.addVar(vtype=grb.GRB.BINARY,
                                  name='x_%d,%d' % (i, j))
             for i in range(n_samples)
             for j in filtered_strategies[i]}
        y = {j: model.addVar(vtype=grb.GRB.BINARY,
                             name='y_%d' % j)
             for j in range(n_strategies)}

        # Constraints
        for i in range(n_samples):
            model.addConstr(grb.quicksum(x[i, j]
                                         for j in filtered_strategies[i]) == 1)
        for i in range(n_samples):
            for j in filtered_strategies[i]:
                model.addConstr(x[i, j] <= y[j])
        model.addConstr(grb.quicksum(y[j] for j in range(n_strategies))
                        <= k_max_strategies)

        for j in range(n_strategies):
            model.addConstr(y[j] <=
                            M[j] * grb.quicksum(x[i, j]
                                                for i in filtered_samples[j]))

        # Objective
        model.setObjective(grb.quicksum(c[i, j] * x[i, j]
                                        for i in range(n_samples)
                                        for j in filtered_strategies[i]))

        # Solve
        model.setParam("OutputFlag", 0)
        model.optimize()
        assert model.Status == grb.GRB.OPTIMAL  # Assert optimal solution

        # Get solution
        x_opt = {(i, j): x[i, j].X
                 for i in range(n_samples)
                 for j in filtered_strategies[i]}
        y_opt = np.array([y[j].X for j in range(n_strategies)])
        degradation = 1. / n_samples * np.sum(c[i, j] * x_opt[i, j]
                                              for i in range(n_samples)
                                              for j in filtered_strategies[i])

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
            for j in filtered_strategies[i]:
                if x_opt[i, j] == 1:
                    self.y_train[i] = np.where(chosen_strategies == j)[0][0]
                    break
            if self.y_train[i] == -1:
                raise ValueError("No strategy selected for sample %d" % i)

        return self.y_train, self.encoding
