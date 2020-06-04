import optuna
from mlopt.learners.learner import Learner
import mlopt.learners.xgboost.settings as xgbs
import mlopt.settings as stg
import mlopt.error as e
import time
import copy


class XGBoostObjective(object):
    def __init__(self, dtrain, bounds, n_classes):
        self.bounds = copy.deepcopy(bounds)
        self.n_classes = n_classes
        import xgboost as xgb
        self.xgb = xgb
        self.dtrain = dtrain

    def __call__(self, trial):
        params = xgbs.DEFAULT_PARAMETERS
        params.update({
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'booster': 'gbtree',
            'num_class': self.n_classes,
            'lambda': trial.suggest_float(
                'lambda', *self.bounds['lambda'], log=True),
            'alpha': trial.suggest_float(
                'alpha', *self.bounds['alpha'], log=True),
            'max_depth': trial.suggest_int(
                'max_depth', *self.bounds['max_depth']),
            'eta': trial.suggest_float(
                'eta', *self.bounds['eta'], log=True),
            'gamma': trial.suggest_float(
                'gamma', *self.bounds['gamma'], log=True),
        })
        n_boost_round = trial.suggest_int(
            'n_boost_round', *self.bounds['n_boost_round'])

        pruning_callback = optuna.integration.XGBoostPruningCallback(
            trial, "test-mlogloss")
        history = self.xgb.cv(params, self.dtrain,
                              num_boost_round=n_boost_round,
                              callbacks=[pruning_callback]
                              )

        mean_loss = history["test-mlogloss-mean"].values[-1]
        return mean_loss


class XGBoost(Learner):

    def __init__(self,
                 **options):
        """
        Initialize OptimalTrees class.

        Parameters
        ----------
        options : dict
            Learner options as a dictionary.
        """
        if not XGBoost.is_installed():
            e.value_error("XGBoost not installed")

        import xgboost as xgb
        self.xgb = xgb

        self.name = stg.XGBOOST
        self.n_input = options.pop('n_input')
        self.n_classes = options.pop('n_classes')
        self.options = {}

        self.options['bounds'] = options.pop(
            'bounds', xgbs.DEFAULT_PARAMETER_BOUNDS)

        not_specified_bounds = \
            [x for x in xgbs.DEFAULT_PARAMETER_BOUNDS.keys()
             if x not in self.options['bounds'].keys()]
        for p in not_specified_bounds:  # Assign remaining keys
            self.options['bounds'][p] = xgbs.DEFAULT_PARAMETER_BOUNDS[p]

        self.options['n_folds'] = options.pop('n_folds', xgbs.N_FOLDS)

        # Pick minimum between n_best and n_classes
        self.options['n_best'] = min(options.pop('n_best', stg.N_BEST),
                                     self.n_classes)

        # Pick number of hyperopt_trials
        self.options['n_train_trials'] = options.pop('n_train_trials',
                                                     stg.N_TRAIN_TRIALS)

        # Mute optuna
        optuna.logging.set_verbosity(optuna.logging.INFO)

    @classmethod
    def is_installed(cls):
        try:
            import xgboost
            xgboost
        except ImportError:
            return False
        return True

    def train(self, X, y):

        self.n_train = len(X)
        dtrain = self.xgb.DMatrix(X, label=y)

        stg.logger.info("Train XGBoost")

        start_time = time.time()
        objective = XGBoostObjective(dtrain, self.options['bounds'],
                                     self.n_classes)

        sampler = optuna.samplers.TPESampler(seed=10)  # Deterministic
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
        study = optuna.create_study(sampler=sampler, pruner=pruner,
                                    direction="minimize")
        study.optimize(objective, n_trials=self.options['n_train_trials'],
                       show_progress_bar=True)

        pruned_trials = [t for t in study.trials
                         if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials
                           if t.state == optuna.trial.TrialState.COMPLETE]

        stg.logger.info("Study statistics: ")
        stg.logger.info("  Number of finished trials: %d" % len(study.trials))
        stg.logger.info("  Number of pruned trials: %d" % len(pruned_trials))
        stg.logger.info("  Number of complete trials: %d" %
                        len(complete_trials))

        self.best_params = study.best_trial.params
        stg.logger.info("Best parameters")
        for key, value in self.best_params.items():
            print("    {}: {}".format(key, value))

        # Train again
        stg.logger.info("Train with best parameters")
        params = xgbs.DEFAULT_PARAMETERS
        params.update({k: v for k, v in self.best_params.items()
                       if k != 'n_boost_round'})
        self.bst = self.xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=self.best_params['n_boost_round']
        )

        # Print timing
        end_time = time.time()
        stg.logger.info("Tree training time %.2f" % (end_time - start_time))

    def predict(self, X):
        y = self.bst.predict(self.xgb.DMatrix(X))
        return self.pick_best_class(y, n_best=self.options['n_best'])

    def save(self, file_name):
        self.bst.save_model(file_name + ".json")

    def load(self, file_name):
        self.bst = self.xgb.Booster().load_model(file_name)
