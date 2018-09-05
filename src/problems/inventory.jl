mutable struct Inventory <: OptimizationProblem
    # Data in LP form
    data::ProblemData

    # Additional data to build the problem
    # (independent from the parameters)
    T::Int64  # Horizon
    M::Float64  # Max ordering capacity
    K::Float64  # Fixed ordering cost
    radius::Float64  # Radius for sampling

    # Empty constructor
    Inventory() = new()
end


function populate!(problem::Inventory,
                   theta::DataFrame;
                   bin_vars::Bool=false,
                   show_model::Bool=false)

    # Get data from theta
    @assert size(theta, 1) == 1  # Only one row
    h = theta[:h][1]
    p = theta[:p][1]
    c = theta[:c][1]
    x0 = theta[:x0][1]
    d = [theta[4+i][1] for i = 1:problem.T]

    T, M, K = problem.T, problem.M, problem.K

    # Define JuMP model
    m = Model(solver=MyModule.BUILD_SOLVER)

    # Variables
    @variable(m, x[i=1:T+1])
    @variable(m, u[i=1:T])
    @variable(m, y[i=1:T])  # Auxiliary: y[t] = max{h * x[t], -p * x[t]}
    (bin_vars) && (@variable(m, v[i=1:T], Bin))

    # Constraints
    @constraint(m, [i=1:length(x0)], x[i] == x0[i])
    @constraint(m, yh[t=1:T], y[t] >= h * x[t])
    @constraint(m, yp[t=1:T], y[t] >= -p * x[t])
    @constraint(m, evolution[t=1:T], x[t + 1] == x[t] + u[t] - d[t])
    @constraint(m, [t=1:T], u[t] >= 0)
    if bin_vars
        @constraint(m, [t=1:T], u[t] <= M * v[t])
    else
        @constraint(m, [t=1:T], u[t] <= M)
    end

    # Cost
    if bin_vars
        @objective(m, Min, sum(y[i] + c * u[i] + K * v[i]  for i in 1:T))
    else
        @objective(m, Min, sum(y[i] + c * u[i] for i in 1:T))
    end

    if show_model
        print(m)
    end

    # Extract problem data
    problem.data = extract_problem_data(m)

end



function sample(problem::Inventory,
                theta_bar::Vector{Float64};
                N=100)

    # Get sampler
    d = MvBall(length(theta_bar), problem.radius, theta_bar)

    # Sample uniform point on the ball
    X = rand(d, N)

    # Construct and return dataframe
    d = DataFrame()
    d[:h] = X[1, :]
    d[:p] = X[2, :]
    d[:c] = X[3, :]
    d[:x0] = X[4, :]
    [d[Symbol("d$i")] = X[4+i, :] for i in 1:problem.T]

    return d



end

