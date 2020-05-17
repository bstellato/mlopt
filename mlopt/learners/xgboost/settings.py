# Parameters
# https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst


DEFAULT_TRAINING_PARAMS = {
    'max_depth': [1, 5, 10],
    'learning_rate' : [0.1, 1],
    'n_estimators': [10, 100],
    'random_state': [0],
    'objective':['multi:softmax'],
    # https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html
    #  'enable_experimental_json_serialization': [True]
    }


N_FOLDS = 5
