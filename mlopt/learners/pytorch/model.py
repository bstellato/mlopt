import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    PyTorch internal neural network class.
    """

    def __init__(self, n_input, n_classes, n_layers, n_hidden):
        super(Net, self).__init__()

        print(n_layers)

        self.in_layer = nn.Linear(n_input, n_hidden)
        self.batchnorm = nn.BatchNorm1d(n_hidden)
        self.linear = nn.ModuleList([nn.Linear(n_hidden, n_hidden)])
        self.linear.extend([nn.Linear(n_hidden, n_hidden)
                            for _ in range(n_layers - 1)])
        self.out_layer = nn.Linear(n_hidden, n_classes)

        # OLD Structure with linear layers
        # self.layers = nn.ModuleList([nn.Linear(n_input, n_hidden)])
        # self.layers.extend([nn.Linear(n_hidden, n_hidden)
        #                     for _ in range(n_layers - 2)])
        # self.layers.append(nn.Linear(n_hidden, n_classes))

    def forward(self, x):

        x = F.relu(self.in_layer(x))
        x = self.batchnorm(x)
        for layer in self.linear:
            x = F.relu(layer(x))
        x = self.out_layer(x)

        # OLD
        # for layer in self.layers[:-1]:
        #     x = F.relu(layer(x))
        # x = self.layers[-1](x)

        return x
