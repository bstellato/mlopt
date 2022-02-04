# Machine Learning Optimizer

`mlopt` is a package to learn how to solve numerical optimization problems from data. It relies on [cvxpy](https://cvxpy.org) for modeling and [gurobi](https://www.gurobi.com/) for solving the problem offline.

`mlopt` learns how to solve programs using [pytorch](https://pytorch.org/) ([pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)), [xgboost](https://xgboost.readthedocs.io/en/latest/) or [optimaltrees](https://docs.interpretable.ai/stable). The machine learning hyperparameter optimization is performed using [optuna](https://optuna.org/).

Online, `mlopt` only requires to predict the strategy and solve a linear system using [scikit-umfpack](https://github.com/scikit-umfpack/scikit-umfpack).

## Examples

To see `mlopt` in action, have a look at the notebooks in the [examples/](./examples/) folder.

## Documentation

Coming soon at [mlopt.org](https://mlopt.org)!

## Citing

If you use `mlopt` for research, please cite the following papers:

* [The Voice of Optimization](https://arxiv.org/pdf/1812.09991.pdf):

  ```
  @Article{bertsimas2021,
  author        = {{Bertsimas}, D. and {Stellato}, B.},
  title         = {The Voice of Optimization},
  journal       = {Machine Learning},
  year          = {2021},
  month         = {2},
  volume        = {110},
  issue         = {2},
  pages         = {249--277},
  }
  ```

* [Online Mixed-Integer Optimization in Milliseconds](https://arxiv.org/pdf/1907.02206.pdf)

  ```
  @article{stellato2019a,
    author = {{Bertsimas}, D. and {Stellato}, B.},
    title = {Online Mixed-Integer Optimization in Milliseconds},
    journal = {arXiv e-prints},
    year = {2019},
    month = jul,
    adsnote = {Provided by the SAO/NASA Astrophysics Data System},
    adsurl = {https://ui.adsabs.harvard.edu/abs/2019arXiv190702206B},
    archiveprefix = {arXiv},
    eprint = {1907.02206},
    keywords = {Mathematics - Optimization and Control},
    pdf = {https://arxiv.org/pdf/1907.02206.pdf},
    primaryclass = {math.OC},
  }

  ```


The code to **reproduce the results in the papers** is available at [bstellato/mlopt_benchmarks](https://github.com/bstellato/mlopt_benchmarks).


## Projects using mlopt framework


* [Learning Mixed-Integer Convex Optimization Strategies for Robot Planning and Control](https://arxiv.org/pdf/2004.03736.pdf)

