import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import logging


# How often should we compute the metrics
METRICS_STEPS = 100


def accuracy_onehot(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) output of the model
        labels: (np.ndarray) batch labels

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels) / float(labels.size)


def mean_squared_error(outputs, labels):
    """
    Compute the mean squared error after rounding, 
    given the outputs and labels.

    Args:
        outputs: (np.ndarray) output of the model
        labels: (np.ndarray) batch labels

    Returns: (float) accuracy in [0,1]
    """
    n_samples = len(labels)
    
    # NB. There should be no need to square since the values can be
    # either 1 or 0
    normalized_outputs = np.reciprocal(1 + np.exp(-outputs))  # Normalize using sigmoid
    differences = np.round(normalized_outputs) - labels
    squared_diff = differences.dot(differences.T)
    errors = np.diag(squared_diff) if n_samples > 1 else squared_diff[0][0]
    mse = np.sum(errors) / n_samples

    return mse
     

def log_metrics(metrics, string="Train"):
    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in metrics])
                    for metric in metrics[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- {} metrics: ".format(string) + metrics_string)

    return metrics_mean


def eval_metrics(outputs, labels, metrics, loss):
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    # compute all metrics on this batch
    summary = {metric: metrics[metric](outputs,
                                       labels)
               for metric in metrics}
    summary['loss'] = loss.item()

    return summary


def get_dataloader(X, y, batch_size=1, ytype=torch.long):
    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=ytype)

    return DataLoader(TensorDataset(X, y),
                      batch_size=batch_size,
                      shuffle=False,
                      )


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg()  # Returns 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0.0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total/float(self.steps)



