# MLOPT Knapsack Example


```python
import numpy as np
import cvxpy as cp
import pandas as pd
import logging

import mlopt
from mlopt.sampling import uniform_sphere_sample
from mlopt.learners.pytorch.pytorch import PytorchNeuralNet
from mlopt.utils import n_features, pandas2array
```

## Generate problem data


```python
np.random.seed(1)  # Reset random seed for reproducibility

# Variable
n = 10
x = cp.Variable(n, integer=True)

# Cost
c = np.random.rand(n)

# Weights
a = cp.Parameter(n, nonneg=True, name='a')
x_u = cp.Parameter(n, nonneg=True, name='x_u')
b = 0.5 * n
```

## Create optimizer object


```python
# Problem
cost = - c * x
constraints = [a * x <= b,
               0 <= x, x <= x_u]


# Define optimizer
# If you just want to remove too many messages
# change INFO to WARNING
m = mlopt.Optimizer(cp.Minimize(cost), constraints,
                    log_level=logging.INFO)
```

## Define training and testing parameters


```python
# Average request
theta_bar = 2 * np.ones(2 * n)
radius = 1.0


def sample(theta_bar, radius, n=100):

    # Sample points from multivariate ball
    ndim = int(len(theta_bar)/2)
    X_a = uniform_sphere_sample(theta_bar[:ndim], radius, n=n)
    X_u = uniform_sphere_sample(theta_bar[ndim:], radius, n=n)

    df = pd.DataFrame({
        'a': list(X_a),
        'x_u': list(X_u)
        })

    return df


# Training and testing data
n_train = 1000
n_test = 100
theta_train = sample(theta_bar, radius, n=n_train)
theta_test = sample(theta_bar, radius, n=n_test)
```

## Train predictor (Pytorch)


```python
# Dictionary of different parameters.
# The cross validation will try all of the possible
# combinations
params = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [32],
    'n_epochs': [10]
}
m.train(theta_train, learner=mlopt.PYTORCH, params=params)
```

## Benchmark on testing dataset


```python
results = m.performance(theta_test)
print("Accuracy: %.2f " % results[0]['accuracy'])
```

## Save training data


```python
m.save_training_data("training_data.pkl", delete_existing=True)
```

## Create new solver and train passing loaded data


```python
m = mlopt.Optimizer(cp.Minimize(cost), constraints)
m.load_training_data("training_data.pkl")
m.train(learner=mlopt.PYTORCH, params=params)  # Train after loading samples

results = m.performance(theta_test)
print("Accuracy: %.2f " % results[0]['accuracy'])
```

## Predict single point


```python
# Predict single point
theta = theta_test.iloc[0]
root = logging.getLogger('mlopt')
root.setLevel(logging.DEBUG)
result_single_point = m.solve(theta)
print(result_single_point)
```

## Learn directly from points (talk directly to pytorch)


```python
y = m.y_train
X = m.X_train
learner = PytorchNeuralNet(n_input=n_features(X),
                           n_classes=len(np.unique(y)),
                           n_best=3,
                           params=params)
# Train learner
learner.train(pandas2array(X), y)

# Predict
X_pred = X.iloc[0]
y_pred = learner.predict(pandas2array(X_pred))  # n_best most likely classes
```


```python

```
