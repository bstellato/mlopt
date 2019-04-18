from mlopt.learners.learner import Learner
from mlopt.settings import N_BEST, FRAC_TRAIN, PYTORCH, \
        NET_TRAINING_PARAMS, DIVISION_TOL
from mlopt.utils import pandas2array
from tqdm import trange
import os
import torch                                            # Basic utilities
import torch.nn as nn                                   # Neural network tools
import torch.nn.functional as F                         # nonlinearitis
import torch.optim as optim                             # Optimizer tools
from torch.utils.data import TensorDataset, DataLoader  # Data manipulaton
import numpy as np
import logging


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        x = 0.5
        nn.init.uniform_(m.bias, -x, x)


class Net(nn.Module):
    """
    PyTorch internal neural network class.
    """

    def __init__(self, n_input, n_classes, n_layers, n_hidden):
        super(Net, self).__init__()
        
        self.layers = nn.ModuleList([nn.Linear(n_input, n_hidden)])
        self.layers.extend([nn.Linear(n_hidden, n_hidden)
                            for _ in range(n_layers - 2)])
        self.layers.append(nn.Linear(n_hidden, n_classes))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)

        return x


class PyTorchNeuralNet(Learner):
    """
    PyTorch Neural Network learner.
    """

    def __init__(self, **options):
        """
        Initialize PyTorch neural network class.

        Parameters
        ----------
        options : dict
            Learner options as a dictionary.
        """
        # Define learner name
        self.name = PYTORCH
        self.n_input = options.pop('n_input')
        self.n_classes = options.pop('n_classes')

        # Default params grid
        default_params = NET_TRAINING_PARAMS

        # Unpack settings
        self.options = {}
        self.options['params'] = options.pop('params', default_params)
        not_specified_params = [x for x in default_params.keys()
                                if x not in self.options['params'].keys()]
        # Assign remaining keys
        for p in not_specified_params:
            self.options['params'][p] = default_params[p]
        if 'n_hidden' not in self.options['params'].keys():
            self.options['params']['n_hidden'] = \
                [int((self.n_classes + self.n_input) / 2)]

        # Pick minimum between n_best and n_classes
        self.options['n_best'] = min(options.pop('n_best', N_BEST),
                                     self.n_classes)
        # Get fraction between training and validation
        self.options['frac_train'] = options.pop('frac_train', FRAC_TRAIN)

        # Define device
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            logging.info("Using CUDA GPU with Pytorch")
        else:
            self.device = torch.device("cpu")
            logging.info("Using CPU with Pytorch")

        # Pick minimum between n_best and n_classes
        self.options['n_best'] = min(options.pop('n_best', N_BEST),
                                     self.n_classes)

        # Define criterion
        self.criterion = nn.CrossEntropyLoss()

    def train_instance(self, X, y, params):
        """
        Train single instance of the network for parameters in params

        params is a dictionary containing
        - batch size
        - n_epochs
        - learning_rate
        - n_layers
        - n_hidden
        """

        # Create PyTorch Neural Network and port to to device
        self.net = Net(self.n_input,
                       self.n_classes,
                       params['n_layers'],
                       params['n_hidden']).to(self.device)

        # Set network in training mode (not evaluation)
        self.net.train()

        info_str = "Learning Neural Network with parameters: "
        info_str += str(params)
        logging.info(info_str)

        # Define optimizer
        #  self.optimizer = optim.SGD(self.net.parameters(),
        #                             lr=params['learning_rate'],
        #                             momentum=0.9)
        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=params['learning_rate'])

        # Convert data to tensor dataset
        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X, y)

        # Define loader for batches
        data_loader = DataLoader(dataset,
                                 batch_size=params['batch_size']
                                 #  shuffle=True
                                 )

        n_batches_per_epoch = \
            int(self.n_train / params['batch_size'])

        # Reset parameters
        torch.manual_seed(1)  # Reset seed
        self.net.apply(weights_init)  # Initialize weights

        with trange(params['n_epochs'], desc="Training neural net") as t:
            for epoch in t:  # loop over dataset multiple times

                avg_cost = 0.0
                for i, (inputs, labels) in enumerate(data_loader):
                    inputs, labels = \
                        inputs.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()                   # zero grad
                    outputs = self.net(inputs)                   # forward
                    self.loss = self.criterion(outputs, labels)  # loss
                    self.loss.backward()                         # backward
                    self.optimizer.step()                        # optimizer

                    avg_cost += self.loss.item() / n_batches_per_epoch

                t.set_description("Training neural net (epoch %4i, cost %.2e)"
                                  % (epoch + 1, avg_cost))

    def normalize(self, X, recompute_stats=True):
        """Normalize data. Recompute mean and std if training mode."""
        if recompute_stats:
            self.X_mean, self.X_std = X.mean(axis=0), X.std(axis=0)

            # Ignore small values of standard deviation
            for i in range(len(self.X_std)):
                if self.X_std[i] < DIVISION_TOL:
                    self.X_std[i] = 1.

        X[:] = (X - self.X_mean) / self.X_std

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

        # Convert X dataframe to numpy array
        X = pandas2array(X)

        # Normalize data
        self.normalize(X, recompute_stats=True)

        # Split dataset in training and validation
        frac_train = self.options['frac_train']
        n_frac_train = int(frac_train * self.n_train)
        n_frac_valid = self.n_train - n_frac_train
        X_train = X[:n_frac_train, :]
        y_train = y[:n_frac_train]
        X_valid = X[n_frac_train:, :]
        y_valid = y[n_frac_train:]
        logging.info("Split dataset in %d training and %d validation" %
                     (n_frac_train, n_frac_valid))

        # Create parameter vector
        params = [{
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'n_layers': n_layers,
            'n_hidden': n_hidden}
            for learning_rate in self.options['params']['learning_rate']
            for batch_size in self.options['params']['batch_size']
            for n_epochs in self.options['params']['n_epochs']
            for n_layers in self.options['params']['n_layers']
            for n_hidden in self.options['params']['n_hidden']]
        n_models = len(params)

        logging.info("Train Neural Network with %d sets of parameters"
                     % n_models)

        # Create vector of results
        accuracy_vec = np.zeros(n_models)

        if n_models > 1:
            for i in range(n_models):

                # Rain with parameters
                self.train_instance(X_train, y_train, params[i])

                # Predict validation (data already normalized)
                y_pred = self.predict(X_valid, n_best=1, normalize=False)

                # Get accuracy
                accuracy_vec[i] = np.sum(
                    np.equal(y_pred.flatten(), y_valid)) / len(y_valid)
                logging.info("Accuracy: %.2f%%" % (accuracy_vec[i] * 100))

            # Pick best parameters
            self.best_params = params[np.argmax(accuracy_vec)]
            logging.info("Best parameters")
            logging.info(str(self.best_params))
            logging.info("Train neural network with best parameters")

        else:

            logging.info("Train neural network with "
                         "just one set of parameters")
            self.best_params = params[0]

        logging.info(self.best_params)
        # Retrain network with best parameters over whole dataset
        self.train_instance(X, y, self.best_params)

    def predict(self, X, n_best=None, normalize=True):

        # Convert X dataframe to numpy array
        X = pandas2array(X)

        # Normalize data
        if normalize:
            self.normalize(X, recompute_stats=False)

        self.net.eval()

        with torch.no_grad():
            n_best = n_best if (n_best is not None) else self.options['n_best']

            # Convert pandas df to array (unroll tuples)
            X = torch.tensor(pandas2array(X), dtype=torch.float)
            X = X.to(self.device)

            # Evaluate probabilities
            # TODO: Required? Maybe we do not need softmax
            y = F.softmax(self.net(X),
                          dim=1).detach().cpu().numpy()

        return self.pick_best_probabilities(y, n_best=n_best)

    def save(self, file_name):
        # Save state dictionary to file
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        torch.save(self.net.state_dict(), file_name + ".pkl")

    def load(self, file_name):
        # Check if file name exists
        if not os.path.isfile(file_name + ".pkl"):
            err = "PyTorch pkl file does not exist."
            logging.error(err)
            raise ValueError(err)

        # Load state dictionary from file
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        self.net.load_state_dict(torch.load(file_name + ".pkl"))
        self.net.eval()  # Necessary to set the model to evaluation mode
