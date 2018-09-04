mutable struct NetlibLP <: OptimizationProblem
    # Data in LP form
    data::ProblemData
    data_orig::ProblemData

    # Additional data to build the problem
    # (independent from the parameters)
    file_name::String
    to_read::Bool

    # Empty constructor
    NetlibLP() = new()
end


function populate!(problem::NetlibLP)

    if problem.to_read
        # Extract nominal problem data
        problem.data_orig = extract_problem_data(problem.file_name)
        problem.data = copy(problem.data_orig)
        problem.to_read = false
    else
        copy!(problem.data, problem.data_origin)
    end

end


function populate!(problem::NetlibLP, theta::DataFrame)
    @assert size(theta, 1) == 1  # Only one row in dataframe
    theta_vec = Array(theta[1, :])[:]

    if problem.to_read
        # Extract nominal problem data if not defined
        problem.data_orig = extract_problem_data(problem.file_name)
        problem.data = copy(problem.data_orig)
        problem.to_read = false
    else
        copy!(problem.data, problem.data_orig)
    end

    # Get problem dimensions
    n_var = length(problem.data.c)
    n_constr = length(problem.data.l)

    # Populate prblem data using theta
    problem.data.c += theta_vec[1:n_var]
    problem.data.l += theta_vec[n_var+1:n_var+n_constr]
    problem.data.u += theta_vec[n_var+n_constr+1:end]

end


"""
    operation_points(problem[, N=10])

Sample the operation points by perturbing the original problem data.
"""
function operation_points(problem::NetlibLP;
                          N::Int64=10)
    perturb_frac = 0.01

    # Problem data
    c, l, u = problem.data.c, problem.data.l, problem.data.u
    n_var = length(c)
    n_constr = length(l)

    # Get equalities and inequalities
    idx_eq, idx_ineq = eq_ineq(problem.data)
    n_eq, n_ineq = length(idx_eq), length(idx_ineq)
    l_eq, u_eq = l[idx_eq], u[idx_eq]
    l_ineq, u_ineq = l[idx_ineq], u[idx_ineq]

    # Get theta vector as
    # c_theta = c + theta_c
    # l_theta = l + theta_l
    # u_theta = u + theta_u

    # Perturb cost
    theta_c = [(perturb_frac * norm(c)) * rand(n_var) for i = 1:N]

    # Do not perturb constraints
    theta_l = [zeros(n_constr) for _ = 1:N]
    theta_u = [zeros(n_constr) for _ = 1:N]

    return [[theta_c[i]; theta_l[i]; theta_u[i]] for i = 1:N]


    #  c_vec = [c for _ in 1:N]
    #  l_vec = [l for _ in 1:N]
    #  u_vec = [u for _ in 1:N]
    #
    # Perturb cost
    #  c_up = c + perturb_frac * abs.(c)
    #  c_low = c # - perturb_frac * abs.(c)
    #  c_vec = [c_low + (c_up - c_low) .* rand(n_var) for _ in 1:N]

    #
    #  # Perturb equalities
    #  delta = perturb_frac * abs.(l_eq)
    #
    #  #DEBUG
    #  delta = zeros(n_eq)
    #
    #  l_eq_vec = [l_eq + delta .* rand(n_eq) - .5 .* delta for _ in 1:N]
    #  u_eq_vec = l_eq_vec
    #
    #  # Perturb inequalities
    #  # Lower
    #  l_up_ineq = l_ineq
    #  l_low_ineq = l_ineq - perturb_frac * abs.(l_ineq)
    #  delta = l_up_ineq - l_low_ineq
    #  [delta[i] = 0 for i = 1:n_ineq if Base.isinf(l_ineq[i])]   # Interval 0 if delta Inf
    #
    #  #DEBUG
    #  delta = zeros(n_ineq)
    #
    #  l_ineq_vec = [l_up_ineq - delta .* rand(n_ineq) for _ in 1:N]
    #
    #  # Upper
    #  u_up_ineq = u_ineq + perturb_frac * abs.(u_ineq)
    #  u_low_ineq = u_ineq
    #  delta = u_up_ineq - u_low_ineq
    #  [delta[i] = 0 for i = 1:n_ineq if Base.isinf(u_ineq[i])]   # Interval 0 if delta Inf
    #
    #  delta = zeros(n_ineq)
    #
    #  u_ineq_vec = [u_low_ineq + delta .* rand(n_ineq) for _ in 1:N]
    #
    #  # Stack result
    #  l_vec = [[l_eq_vec[i]; l_ineq_vec[i]] for i in 1:N]
    #  u_vec = [[u_eq_vec[i]; u_ineq_vec[i]] for i in 1:N]
    #
    #  # Check if some problems get infeasible
    #  for i = 1:N
    #      if any(l_vec[i] .> u_vec[i])
    #          println("Error violation in $(i)")
    #      else
    #          println("All is OK")
    #      end
    #
    #      if norm(l_vec[i] - l) >= 1e-05
    #          println("Wrong l vector $(i)")
    #      end
    #
    #      if norm(u_vec[i] - u) >= 1e-05
    #          println("Wrong u vector $(i)")
    #      end
    #  end
    #
    #  @show [length(l) for l in l_vec]
    #  @show [length(u) for u in u_vec]
    #  @show n_eq + n_ineq

    # Perturb lower bound
    #  l_up = l
    #  l_low = l - perturb_frac * abs.(l)
    #  delta = l_up - l_low
    #  [delta[i] = 0 for i = 1:n_constr if Base.isinf(l[i])]   # Interval 0 if delta Inf
    #  l_vec = [l_low + delta .* rand(n_constr) for _ in 1:N]

    # Perturb upper bound
    #  u_up = u + perturb_frac * abs.(u)
    #  u_low = u
    #  delta = u_up - u_low
    #  [delta[i] = 0 for i = 1:n_constr if Base.isinf(u[i])]   # Interval 0 if delta Inf
    #  u_vec = [u_low + delta .* rand(n_constr) for _ in 1:N]
    #
    #  @show sum(abs.(u - l) .<= 1e-03)

    # Stack points up
    #  temp = [[c_vec[i]; l_vec[i]; u_vec[i]] for i = 1:N]

    #  [@show size(t) for t in temp]
end


function sample(problem::NetlibLP,
                theta_bar::Vector{Vector{Float64}},
                r::Float64;
                N=100)

    n_var = length(problem.data.c)
    n_constr = length(problem.data.l)

    # Get sampling points per operation point
    n_op = length(theta_bar)  # Numer of operation points
    n_sample_per_op = floor(Int, N / n_op)   # Number of samples per operation point

    # TODO: change this
    idx_sampling = find(abs.(theta_bar[1]) .> TOL)
    n_sampling = length(idx_sampling)

    # Get values that are not infinity (from first element)
    #  idx_finite = find(.!Base.isinf.(theta_bar[1]))

    # TODO: Get index that are not equalities

    #  n_finite = length(idx_finite)

    # Get sampling points per operation point
    theta = Vector{Vector{Float64}}(0)
    for k = 1:length(theta_bar)
        # Get sampler around the non infinity
        # elements of theta_bar[k]
        d = MvBall(n_sampling, r, theta_bar[k][idx_sampling])

        # Sample uniform point on the ball
        theta_new = [copy(theta_bar[k]) for _ in 1:n_sample_per_op]
        [theta_new[i][idx_sampling] = rand(d) for i in 1:n_sample_per_op]

        # Add sample to list of points
        append!(theta, theta_new)
    end

    # Create dataframe
    X = vcat(theta'...)
    d = DataFrame()
    [d[Symbol("c$i")] = X[:, i] for i in 1:n_var]
    [d[Symbol("l$i")] = X[:, n_var + i] for i in 1:n_constr]
    [d[Symbol("u$i")] = X[:, n_var + n_constr + i] for i in 1:n_constr]

    return d

end

