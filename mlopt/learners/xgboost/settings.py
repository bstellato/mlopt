# Parameters
# https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst

#  DEFAULT_TRAINING_PARAMS = {
#      'max_depth': [1, 5, 10],
#      'learning_rate' : [0.1, 1],
#      'n_estimators': [10, 100],
#      'random_state': [0],
#      'objective':['multi:softmax'],
#      # https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html
#      #  'enable_experimental_json_serialization': [True]
#      }

DEFAULT_PARAMETERS = {
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'booster': 'gbtree',
}

DEFAULT_PARAMETER_BOUNDS = {
    'lambda': [1e-8, 2.0],
    'alpha': [1e-9, 2.0],
    'max_depth': [1, 15],
    'eta': [1e-9, 2.0],
    'gamma': [1e-8, 2.0],
    'n_boost_round': [50, 400],
}


N_FOLDS = 5
