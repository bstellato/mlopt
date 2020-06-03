from mlopt.learners.learner import Learner
import mlopt.learners.xgboost.settings as xgbstg
import mlopt.settings as stg
from mlopt.utils import get_n_processes
import mlopt.error as e
import numpy as np
import os
import time


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
            e.error("XGBoost not installed")

        # Import julia and IAI module
        from xgboost.sklearn import XGBClassifier
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        self.XGBClassifier = XGBClassifier
        self.GridSearchCV = GridSearchCV
        self.StratifiedKFold = StratifiedKFold
        self.name = stg.XGBOOST
        self.n_input = options.pop('n_input')
        self.n_classes = options.pop('n_classes')

        self.options = {}
        self.options['params'] = options.pop('params', xgbstg.DEFAULT_TRAINING_PARAMS)
        not_specified_params = [x for x in xgbstg.DEFAULT_TRAINING_PARAMS.keys()
                                if x not in self.options['params'].keys()]
        for p in not_specified_params:  # Assign remaining keys
            self.options['params'][p] = xgbstg.DEFAULT_TRAINING_PARAMS[p]

        self.options['n_folds'] = options.pop('n_folds', xgbstg.N_FOLDS)

        # Pick minimum between n_best and n_classes
        self.options['n_best'] = min(options.pop('n_best', stg.N_BEST),
                                     self.n_classes)


    @classmethod
    def is_installed(cls):
        try:
            import xgboost
            import sklearn
        except ImportError:
            return False
        return True

    def train(self, X, y):

        self.n_train = len(X)

        stg.logger.info("Train XGBoost")

        start_time = time.time()
        clf = self.GridSearchCV(self.XGBClassifier(random_state=0),
                                self.options['params'],
                                cv=self.StratifiedKFold(n_splits=self.options['n_folds'],
                                                        shuffle=True, random_state=0),
                                scoring='accuracy',
                                verbose=3,
                                refit=True)
        clf.fit(X, y)

        self.best_params = clf.best_params_
        self.bst = clf.best_estimator_  # Store boosted tree
        stg.logger.info("Best parameters")
        stg.logger.info(str(self.best_params))

        # Print timing
        end_time = time.time()
        stg.logger.info("Tree training time %.2f" % (end_time - start_time))

    def predict(self, X):
        y = self.bst.predict_proba(X)
        return self.pick_best_class(y, n_best=self.options['n_best'])

    def save(self, file_name):
        self.bst.save_model(file_name + ".json")

    def load(self, file_name):
        self.bst = self.XGBClassifier(random_state=0).load(file_name + ".json")
