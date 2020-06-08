# Machine Learning Optimizer

`mlopt` is a package to learn how to solve numerical optimization problems from data. It relies on [CVXPY](https://cvxpy.org) for modeling and [GUROBI](https://www.gurobi.com/) for solving the problem offline.

`mlopt` learns how to solve programs using [Pytorch](https://pytorch.org/), [XGBoost](https://xgboost.readthedocs.io/en/latest/) or [OptimalTrees](https://docs.interpretable.ai/stable). The machine learning hyperparameter optimization is performed using [optuna](https://optuna.org/).

Online, `mlopt` only requires to predict the strategy and solve a linear system using [scikit-umfpack](https://github.com/scikit-umfpack/scikit-umfpack).

## Examples

To see `mlopt` in action, have a look at the notebooks in the [examples/](./examples/) folder.

## Documentation

Coming soon at [mlopt.org](https://mlopt.org)!

## Citing

If you use `mlopt` for research, please cite the following papers:

* [The Voice of Optimization](https://arxiv.org/pdf/1812.09991.pdf):

  ```
  @article{mlopt,
    author = {{Bertsimas}, D. and {Stellato}, B.},
    title = {The Voice of Optimization},
    journal = {Machine Learning (to appear)},
    year = {2020},
    month = jun,
  }
  ```

* [Online Mixed-Integer Optimization in Milliseconds](https://arxiv.org/pdf/1907.02206.pdf)

  ```
  @article{stellato2019a,
    author = {{Bertsimas}, D. and {Stellato}, B.},
    title = {Online Mixed-Integer Optimization in Milliseconds},
    journal = {INFORMS Journal on Computing (major revision)},
    year = {2019},
    month = jul,
  }
  ```


## Projects using mlopt framework


* [Learning Mixed-Integer Convex Optimization Strategies for Robot Planning and Control](https://arxiv.org/pdf/2004.03736.pdf)

