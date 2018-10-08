from .learner import Learner
import numpy as np
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

    def __init__(self, n_input, n_layers, n_classes):
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

    def __enter__(self):
        """Enter for context manager"""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit for context manager"""
        pass

    def __init__(self,
                 n_input,
                 n_layers,
                 n_classes,
                 n_epochs=1000,
                 learning_rate=0.001,
                 momentum=0.9,
                 batch_size=100):

        # Assign settings
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_input = n_input
        self.n_classes = n_classes

        # Define device
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
            )

        # Create PyTorch Neural Network and port to to device
        self.net = Net(n_input, n_layers, n_classes).to(self.device)

        # Define criterion
        self.criterion = nn.CrossEntropyLoss()

        # Define optimizer
        #  self.optimizer = optim.SGD(self.net.parameters(),
        #                             lr=learning_rate,
        #                             momentum=momentum)
        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=learning_rate)

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

        # Convert data to tensor dataset
        X = torch.tensor(self.pandas2array(X), dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X, y)

        # Define loader for batches
        data_loader = DataLoader(dataset, batch_size=self.batch_size,
                                 shuffle=True)

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

        return self.predict_best(X, k=1)

    def predict_best(self, X, k=1):

        # Count number of points
        n_points = len(X)

        # Convert pandas df to array (unroll tuples)
        X = torch.tensor(self.pandas2array(X), dtype=torch.float)
        X.to(self.device)

        # Evaluate probabilities
        y_pred = F.softmax(self.net(X),
                           dim=1).detach().numpy()

        # Sort probabilities
        idx_probs = np.empty((n_points, k), dtype='int')
        for i in range(n_points):
            # Get best k indices
            # NB. Argsort sorts in reverse mode
            idx_probs[i, :] = np.argsort(y_pred[i, :])[-k:]

        return idx_probs

        #  # Predict using internal model with data X
        #  # Test model
        #  correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        #  # Calculate accuracy
        #  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #  print("Accuracy:", accuracy.eval({x: mnist.test.images,
        #                                    y: mnist.test.labels}))
        #
