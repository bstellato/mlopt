mutable struct Assignment <: OptimizationProblem
    # Data in LP form
    data::ProblemData

    # Additional data to build the problem
    # (independent from the parameters)
    A::Int64

    # Empty constructor
    Assignment() = new()
end


function populate!(problem::Assignment, theta::DataFrame)

    @assert size(theta, 1) == 1  # Only one row
    theta_vec = Array(theta)[:]

    # Get dimension
    A = problem.A

    T = A  # Same tasks as agents

    c = eye(A) + spdiagm(theta_vec)

    # Define JuMP model
    m = Model(solver=MyModule.BUILD_SOLVER)

    # Variables
    @variable(m, x[i=1:A, j=1:T] >= 0)

    # Constraints
    @constraint(m, [i=1:A], sum(x[i, j] for j = 1:T) == 1)
    @constraint(m, [j=1:T], sum(x[i, j] for i = 1:A) == 1)

    # Objective
    @objective(m, Min, sum(c[i, j] * x[i, j]  for i in 1:A for j = 1:T))

    # Extract problem data
    problem.data = extract_problem_data(m)

end


function sample(problem::Assignment,
                theta_bar::Vector{Float64},
                r::Float64;
                N=100)

    # Get sampler
    d = MvBall(length(theta_bar), r, theta_bar)

    # Sample uniform point on the ball
    X = rand(d, N)

    # Construct and return dataframe
    return DataFrame(X')

end

