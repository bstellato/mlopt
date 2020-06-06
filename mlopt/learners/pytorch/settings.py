#  DEFAULT_TRAINING_PARAMS = {
#      'learning_rate': [1e-04, 1e-03, 1e-02],
#      'n_epochs': [20],
#      'batch_size': [32],
#      # 'n_layers': [5, 7, 10]
#  }


PARAMETERS = {
    # TOFILL
    #  'objective': 'multi:softprob',
    #  'eval_metric': 'mlogloss',
    #  'booster': 'gbtree',
}

PARAMETER_BOUNDS = {
    'n_layers': [3, 15],
    'n_units_l': [4, 128],
    'learning_rate': [1e-05, 1e1],
    'dropout': [0.1, 0.5],
    'n_epochs': [5, 30],
    'batch_size': [32, 256],
}

N_FOLDS = 5
