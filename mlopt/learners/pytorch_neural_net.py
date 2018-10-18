from mlopt.learners.learner import Learner
from mlopt.settings import N_BEST, PYTORCH
from mlopt.utils import pandas2array
from tqdm import trange
import os
import torch                                            # Basic utilities
import torch.nn as nn                                   # Neural network tools
import torch.nn.functional as F                         # nonlinearitis
import torch.optim as optim                             # Optimizer tools
from torch.utils.data import TensorDataset, DataLoader  # Data manipulaton


class Net(nn.Module):
    """
    PyTorch internal neural network class.
    """

    def __init__(self, n_input, n_classes):
        super(Net, self).__init__()
        n_hidden = int((n_classes + n_input) / 2)
        self.f1 = nn.Linear(n_input, n_hidden)
        self.f2 = nn.Linear(n_hidden, n_hidden)
        self.f3 = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        x = F.relu(self.f1(x))  # First layer
        x = F.relu(self.f2(x))  # Second layer
        x = self.f3(x)          # Third layer
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

        # Unpack settings
        self.options = {}
        self.options['learning_rate'] = options.pop('learning_rate', 0.001)
        self.options['n_epochs'] = options.pop('n_epochs', 1000)
        self.options['batch_size'] = options.pop('batch_size', 100)
        self.n_input = options.pop('n_input')
        self.n_classes = options.pop('n_classes')
        self.options['n_best'] = options.pop('n_best', N_BEST)

        # Reset torch seed
        torch.manual_seed(1)

        # Define device
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        # Create PyTorch Neural Network and port to to device
        self.net = Net(self.n_input,
                       self.n_classes).to(self.device)

        # Define criterion
        self.criterion = nn.CrossEntropyLoss()

        # Define optimizer
        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=self.options['learning_rate'])

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

        self.options['n_train'] = len(X)

        # Convert data to tensor dataset
        X = torch.tensor(pandas2array(X), dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X, y)

        # Define loader for batches
        data_loader = DataLoader(dataset,
                                 batch_size=self.options['batch_size'],
                                 #  shuffle=True
                                 )

        n_batches_per_epoch = \
            int(self.options['n_train'] / self.options['batch_size'])
        with trange(self.options['n_epochs'], desc="Training neural net") as t:
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

        print('Finished training')

    def predict(self, X):

        # Convert pandas df to array (unroll tuples)
        X = torch.tensor(pandas2array(X), dtype=torch.float)
        X.to(self.device)

        # Evaluate probabilities
        # TODO: Required? Maybe we do not need softmax
        y = F.softmax(self.net(X),
                      dim=1).detach().numpy()

        return self.pick_best_probabilities(y)

    def save(self, file_name):
        # Save state dictionary to file
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        torch.save(self.net.state_dict(), file_name + ".pkl")

    def load(self, file_name):
        # Check if file name exists
        if not os.path.isfile(file_name + ".pkl"):
            raise ValueError("PyTorch pkl file does not exist.")

        # Load state dictionary from file
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        self.net.load_state_dict(torch.load(file_name + ".pkl"))
        self.net.eval()  # Necessary to set the model to evaluation mode
