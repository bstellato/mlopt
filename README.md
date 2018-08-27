# Parametric Optimization using Machine Learning


## TODO

- [x] Rewrite functions so that they get `lb` and `ub` so that there is no need to change the input arguments for `b`


### Inventory Control
- OK - Rewrite problem as MathProgBase form: https://mathprogbasejl.readthedocs.io/en/latest/lpqcqp.html
- OK Fix basis computation and solve using the functions in extract_basis.jl and learn_basis.jl
- TODO Finish Inventory control example

- [ ] Try more realistic LP with parameters
  - [ ] Reasonable parameters influence (linear interaction?)
  - [ ] Reasonable size


## Thoughts and IDEAS

### Continuous optimization: Active Sets learning
We learn a representation of the optimal solution. It is not the solution itself but all the information we need to efficiently get it.
This works well for convex optimization problem where the active set are what we only need. Especially if the constraints are linear.

### Integer Optimization: Optimal LP relaxation learning (new stuff!)
This is what we need for convex mixed-integer programs. A generalized concept of the active set, i.e., the information we need to obtain the
optimal solution.

### Robust Optimization: Learn the worst case parameter values
Similar to active set: value of the parameter that is hit when the robust constraint is active.

## Novelties
- Trees as multiclass classifiers. Never used before! Before only ensemble policy. Not really the problem you would like to solve.
- Interpretability of the result
- Do not use distributions and sampling but JUST incoming data (or history).
- Extension to Integer Programs (hard)
- Extension to Robust Programs (hard)
- Extension to multistage optimization
