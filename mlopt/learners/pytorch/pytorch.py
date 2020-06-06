import mlopt.settings as stg
import mlopt.learners.pytorch.settings as pts
from pytorch_lightning.callbacks import PyTorchLightningPruningCallback
from pytorch_lightning import Trainer
from mlopt.learners.pytorch.lightning import LightningNet
from pytorch_lightning import Callback
import mlopt.error as e
from mlopt.learners.learner import Learner
from sklearn.model_selection import train_test_split
import optuna
from time import time


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
        self.n_classes = n_classes
        self.n_input = n_input
        self.data = data

    def __call__(self, trial):

        # The default logger in PyTorch Lightning writes to event files
        # to be consumed by TensorBoard. We don't use any logger here as
        # it requires us to implement several abstract methods. Instead
        # we setup a simple callback, that saves metrics from each
        # validation step.
        metrics_callback = MetricsCallback()

        max_epochs = trial.suggest_int(
            'max_epochs', *self.bounds['max_epochs'])

        trainer = Trainer(
            logger=False,  # ??
            #  val_percent_check=PERCENT_VALID_EXAMPLES, # ??
            #  checkpoint_callback=checkpoint_callback,
            max_epochs=max_epochs,
            gpus=-1 if self.use_gpu else None,
            callbacks=[metrics_callback],
            early_stop_callback=PyTorchLightningPruningCallback(
                trial, monitor="val_loss"),
        )

        model = LightningNet(self.data, self.bounds, self.n_input,
                             self.n_classes, trial)
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
            e.error("Pytorch not installed")

        # import torch
        import torch
        self.torch = torch

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
            stg.logger.info("Using CUDA GPU %s with Pytorch" %
                            self.torch.cuda.get_device_name(self.device))
        else:
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
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y,
                                                              test_size=0.1,
                                                              random_state=0)

        data = {'X_train': X_train, 'y_train': y_train,
                'X_valid': X_valid, 'y_valid': y_valid}
        stg.logger.info("Split dataset in %d training and %d validation" %
                        (len(y_train), len(y_valid)))

        start_time = time.time()
        objective = PytorchObjective(data, self.options['bounds'],
             self.n_input, self.n_classes, self.use_gpu)

        sampler = optuna.samplers.TPESampler(seed=0)  # Deterministic
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
        study = optuna.create_study(sampler=sampler, pruner=pruner,
                                    direction="minimize")
        study.optimize(objective,
                       n_trials=self.options['n_train_trials'],
                       #  show_progress_bar=True
                       )


        # TODO: Continue from here
