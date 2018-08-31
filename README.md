# Parametric Optimization using Machine Learning


## TODO

- [ ] Solve problems netlib lps.
  - [ ] Sample different points for b and c
  - [ ] Sample balls around them
  - [ ] Perform training and testing
- [ ] Develop Inventory model
  - [ ] Increase uncertain parameters
  - [ ] (s, S) policies
- [ ] Investigate quadratic programs
  - [ ] Parametric examples


# Find additional problems
- [ ] Find good list of example problems
  * Boyd book page 148
  * Fitting with l1

- [ ] Minimize convex PWL functions
  * l-infinity norm (Chebyshev) approximatiomn: min || A x - b ||_inf
  * l-1 norm approximation min || A x - b ||_1
  * l-1 curve fitting
  * l-1 signal recovery min || x ||_1 subject to Ax = y


- [ ] Examples (Optimization Methods)
  * Transportation Problem: Max Flow
  * Sorting via LO (Interesting example of fast problem)
  * Investment under taxation
  * Investment problem
  * Manufacturing
  * Scheduling
  * Cutting Stock Problem (Slides lecture 8). Many variables


- [ ] Other examples
  * Network Optimization?
  * Optimal Transportation as LP?

- [ ] LO problems from NETLIB (Gay 1985), all of which are relatively easy for modern solvers in their nominal forms.
- [ ] LO problems from Hans Mittelmann’s benchmark library (Mittelmann 2015).

- [ ] Quadratic Programs (?)


Questions
- [ ] How often are the solutions the same?
- [ ] How infeasible?
- [ ] How suboptimal?
- [ ] How does it scale?



## Random problems
- Get data
- Sample within a radius from each data point

## Filter active sets
- Sort them
- Take most frequent ones
- Find a way to map active sets to close ones
- Sensitivity analysis!

### Inventory Control

- [ ] Try more realistic LP with parameters
  - [ ] Reasonable parameters influence (linear interaction?)
  - [ ] Reasonable size


