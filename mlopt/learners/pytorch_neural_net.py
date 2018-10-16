from mlopt.learners.learner import Learner
from mlopt.settings import N_BEST
from tqdm import trange
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

        # Unpack settings
        self.learning_rate = options.pop('learning_rate', 0.001)
        self.n_epochs = options.pop('n_epochs', 1000)
        self.batch_size = options.pop('batch_size', 100)
        self.n_input = options.pop('n_input')
        self.n_classes = options.pop('n_classes')
        self.n_best = options.pop('n_best', N_BEST)

        # Reset torch seed
        torch.manual_seed(1)

        # Define device
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        # Create PyTorch Neural Network and port to to device
        self.net = Net(self.n_input, self.n_classes).to(self.device)

        # Define criterion
        self.criterion = nn.CrossEntropyLoss()

        # Define optimizer
        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=self.learning_rate)

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

        import ipdb; ipdb.set_trace()
        self.n_train = len(X)

        # Convert data to tensor dataset
        X = torch.tensor(self.pandas2array(X), dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X, y)

        # Define loader for batches
        data_loader = DataLoader(dataset, batch_size=self.batch_size,
                                 #  shuffle=True
                                 )

        n_batches_per_epoch = int(self.n_train / self.batch_size)
        with trange(self.n_epochs, desc="Training neural net") as t:
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
        X = torch.tensor(self.pandas2array(X), dtype=torch.float)
        X.to(self.device)

        # Evaluate probabilities
        # TODO: Required? Maybe we do not need softmax
        y = F.softmax(self.net(X),
                      dim=1).detach().numpy()

        return self.pick_best_probabilities(y)
