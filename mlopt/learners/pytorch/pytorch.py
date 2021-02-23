import mlopt.settings as stg
import mlopt.learners.pytorch.settings as pts
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Trainer
from mlopt.learners.pytorch.lightning import LightningNet
from pytorch_lightning import Callback
import mlopt.error as e
from mlopt.learners.learner import Learner
from sklearn.model_selection import train_test_split
import optuna
from time import time
import os
import logging


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


class PytorchObjective(object):

    def __init__(self, data, bounds, n_input, n_classes, use_gpu=False):
        self.use_gpu = use_gpu
        self.bounds = bounds
        self.data = data
        self.n_input = n_input
        self.n_classes = n_classes

    def __call__(self, trial):

        # The default logger in PyTorch Lightning writes to event files
        # to be consumed by TensorBoard. We don't use any logger here as
        # it requires us to implement several abstract methods. Instead
        # we setup a simple callback, that saves metrics from each
        # validation step.
        metrics_callback = MetricsCallback()

        # Define parameters
        parameters = {
            'n_input': self.n_input,
            'n_classes': self.n_classes,
            'n_layers': trial.suggest_int('n_layers',
                                          *self.bounds['n_layers']),
            'dropout': trial.suggest_uniform('dropout',
                                             *self.bounds['dropout']),
            'batch_size': trial.suggest_int('batch_size',
                                            *self.bounds['batch_size']),
            'learning_rate': trial.suggest_float('learning_rate',
                                                 *self.bounds['learning_rate'],
                                                 log=True),
            'max_epochs': trial.suggest_int('max_epochs',
                                            *self.bounds['max_epochs'])
        }
        for i in range(parameters['n_layers']):
            parameters['n_units_l{}'.format(i)] = trial.suggest_int(
                'n_units_l{}'.format(i), *self.bounds['n_units_l'], log=True)

        # Construct trainer object and train
        trainer = Trainer(
            logger=False,
            checkpoint_callback=False,
            accelerator='dp',
            max_epochs=parameters['max_epochs'],
            gpus=-1 if self.use_gpu else None,
            callbacks=[metrics_callback,
                       PyTorchLightningPruningCallback(trial,
                                                       monitor="val_loss")],

        )

        model = LightningNet(parameters, self.data)
        trainer.fit(model)

        return metrics_callback.metrics[-1]["val_loss"]


class PytorchNeuralNet(Learner):
    """Pytorch Learner class. """

    def __init__(self, **options):
        """Initialize Pytorch Learner class.

        Parameters
        ----------
        options : dict
            Learner options as a dictionary.
        """
        if not PytorchNeuralNet.is_installed():
            e.value_error("Pytorch not installed")

        # import torch
        import torch
        self.torch = torch

        # Disable logging
        log = logging.getLogger("lightning")
        log.setLevel(logging.ERROR)

        self.name = stg.PYTORCH
        self.n_input = options.pop('n_input')
        self.n_classes = options.pop('n_classes')
        self.options = {}

        self.options['bounds'] = options.pop(
            'bounds', pts.PARAMETER_BOUNDS)

        not_specified_bounds = \
            [x for x in pts.PARAMETER_BOUNDS.keys()
             if x not in self.options['bounds'].keys()]
        for p in not_specified_bounds:  # Assign remaining keys
            self.options['bounds'][p] = pts.PARAMETER_BOUNDS[p]

        # Pick minimum between n_best and n_classes
        self.options['n_best'] = min(options.pop('n_best', stg.N_BEST),
                                     self.n_classes)

        # Pick number of hyperopt_trials
        self.options['n_train_trials'] = options.pop('n_train_trials',
                                                     stg.N_TRAIN_TRIALS)

        # Mute optuna
        optuna.logging.set_verbosity(optuna.logging.INFO)

        # Define device
        self.use_gpu = self.torch.cuda.is_available()
        if self.use_gpu:
            self.device = self.torch.device("cuda:0")
            stg.logger.info("Using CUDA GPU %s with Pytorch" %
                            self.torch.cuda.get_device_name(self.device))
        else:
            self.device = self.torch.device("cpu")
            stg.logger.info("Using CPU with Pytorch")

    @classmethod
    def is_installed(cls):
        try:
            import torch
            torch
        except ImportError:
            return False
        return True

    def train(self, X, y):
        """
        Train model.

        Parameters
        ----------
        X : pandas DataFrame
            Features.
        y : numpy int array
            Labels.
        """

        self.n_train = len(X)

        # Split train and validation
        X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                              #  stratify=y,
                                                              test_size=stg.FRAC_TRAIN,
                                                              random_state=0)

        data = {'X_train': X_train, 'y_train': y_train,
                'X_valid': X_valid, 'y_valid': y_valid}
        stg.logger.info("Split dataset in %d training and %d validation" %
                        (len(y_train), len(y_valid)))

        start_time = time()
        objective = PytorchObjective(data, self.options['bounds'],
                                     self.n_input, self.n_classes,
                                     self.use_gpu)

        sampler = optuna.samplers.TPESampler(seed=0)  # Deterministic
        pruner = optuna.pruners.MedianPruner(
            #  n_warmup_steps=5
        )
        study = optuna.create_study(sampler=sampler, pruner=pruner,
                                    direction="minimize")
        study.optimize(objective,
                       n_trials=self.options['n_train_trials'],
                       #  show_progress_bar=True
                       )

        # DEBUG
        #  fig = optuna.visualization.plot_intermediate_values(study)
        #  fig.show()

        self.best_params = study.best_trial.params
        self.best_params['n_input'] = self.n_input
        self.best_params['n_classes'] = self.n_classes

        self.print_trial_stats(study)

        # Train again
        stg.logger.info("Train with best parameters")

        self.trainer = Trainer(
            checkpoint_callback=False,
            accelerator='dp',
            logger=False,  # ??
            max_epochs=self.best_params['max_epochs'],
            gpus=-1 if self.use_gpu else None,
        )
        self.model = LightningNet(self.best_params, data)
        self.trainer.fit(self.model)

        # Print timing
        end_time = time()
        stg.logger.info("Training time %.2f" % (end_time - start_time))

    def predict(self, X):

        # Disable gradients computation
        #  self.model.eval()  # Put layers in evaluation mode
        #  with self.torch.no_grad():  # Needed?
        #
        #      X = self.torch.tensor(X, dtype=self.torch.float).to(self.device)
        #      y = self.model(X).detach().cpu().numpy()

        X = self.torch.tensor(X, dtype=self.torch.float)
        with self.torch.no_grad():
            y = self.model(X).detach().numpy()

        return self.pick_best_class(y, n_best=self.options['n_best'])

    def save(self, file_name):
        self.trainer.save_checkpoint(file_name + ".ckpt")

        # Save state dictionary to file
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        #  self.torch.save(self.model.state_dict(), file_name + ".pkl")

    def load(self, file_name):
        path = file_name + ".ckpt"
        # Check if file name exists
        if not os.path.isfile(path):
            e.value_error("Pytorch checkpoint file does not exist.")

        self.model = LightningNet.load_from_checkpoint(
                checkpoint_path=path, hparams_file=self.best_params
        ).to(self.device)
        self.model.eval()  # Necessary to set the model to evaluation mode
        self.model.freeze()  # Necessary to set the model to evaluation mode

        #  # Load state dictionary from file
        #  # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        #  self.model.load_state_dict(self.torch.load(file_name + ".pkl"))
        #  self.model.eval()  # Necessary to set the model to evaluation mode
