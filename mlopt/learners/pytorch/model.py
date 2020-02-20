import torch.nn as nn
import torch.nn.functional as F


# Neural network default parameters
DEFAULT_TRAINING_PARAMS = {
    'learning_rate': [1e-04, 1e-03, 1e-02],
    'n_epochs': [20],
    'batch_size': [32],
    # 'n_layers': [5, 7, 10]
}


class Net(nn.Module):
    """
    PyTorch internal neural network class.
    """

    def __init__(self, n_input, n_classes, n_hidden):
        super(Net, self).__init__()

        self.in_layer = nn.Linear(n_input, n_hidden)
        self.batchnorm = nn.BatchNorm1d(n_hidden)
        self.linear1 = nn.Linear(n_hidden, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_hidden)
        self.out_layer = nn.Linear(n_hidden, n_classes)

        # OLD Structure with linear layers
        # self.layers = nn.ModuleList([nn.Linear(n_input, n_hidden)])
        # self.layers.extend([nn.Linear(n_hidden, n_hidden)
        #                     for _ in range(n_layers - 2)])
        # self.layers.append(nn.Linear(n_hidden, n_classes))

    def forward(self, x):

        x = F.relu(self.in_layer(x))
        x = self.batchnorm(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.out_layer(x)

        # OLD
        # for layer in self.layers[:-1]:
        #     x = F.relu(layer(x))
        # x = self.layers[-1](x)

        return x
