# Functions to perturb benchmark library LPs

"""
    operation_points(problem[, N=10])

Sample the operation points by perturbing the original problem data.
"""
function operation_points(problem::OptimizationProblem;
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
    theta_c = [perturb_frac * abs.(c) .* rand(n_var) for i = 1:N]

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





# Create multivariate distribution sampler
struct MvBall <: Sampleable{Multivariate, Continuous}
    n::Int64                 # Dimension
    r::Float64               # Radius
    center::Vector{Float64}  # Center of the ball
end
Base.length(s::MvBall) = s.n # return the length of each sample

"""
Generate a single vector sample to x

The function initially samples the points using a Normal Distribution
with `randn`. Then the incomplete gamma function is used to map the
points radially to fit in the hypersphere of finite radius r with
a uniform spatial distribution.

In order to have a uniform distributions over the sphere we multiply the vectors
by f(r): f(r)*r is distributed with density proportional to r^n on [0,1].
"""
function Distributions._rand!{T<:Real}(s::MvBall, x::AbstractVector{T})
    n, r, center = s.n, s.r, s.center
    x_sample = randn(n)
    s2 = norm(x_sample)^2
    # NB. sf_gamma_inc_P(a, x) computes the normalized incomplete gamma function
    # for a and x (the arguments are swapped from the Matlab version).
    fr = r*(sf_gamma_inc_P(n/2, s2/2)^(1/n))/sqrt(s2)
    # Multiply by fr and shift by center
    x_sample = center + fr .* x_sample

    # Assign values of x
    x[:] = x_sample[:]

    #  This is translated from the following Matlab implementation
    #  % This function returns an m by n array, X, in which
    #  % each of the m rows has the n Cartesian coordinates
    #  % of a random point uniformly-distributed over the
    #  % interior of an n-dimensional hypersphere with
    #  % radius r and center at the origin.  The function
    #  % 'randn' is initially used to generate m sets of n
    #  % random variables with independent multivariate
    #  % normal distribution, with mean 0 and variance 1.
    #  % Then the incomplete gamma function, 'gammainc',
    #  % is used to map these points radially to fit in the
    #  % hypersphere of finite radius r with a uniform % spatial distribution.
    #  % Roger Stafford - 12/23/05
    #
    #  X = randn(m,n);
    #  s2 = sum(X.^2,2);
    #  X = X.*repmat(r*(gammainc(s2/2,n/2).^(1/n))./sqrt(s2),1,n);

end



function sample(theta_bar::Vector{Vector{Float64}},
                r::Float64;
                N=100)

    # Get sampling points per operation point
    n_op = length(theta_bar)  # Numer of operation points
    n_sample_per_op = floor(Int, N / n_op)   # Number of samples per operation point

    # Get values that are not infinity (from first element)
    idx_finite = find(.!Base.isinf.(theta_bar[1]))

    # TODO: Get index that are not equalities

    n_finite = length(idx_finite)

    # Get sampling points per operation point
    theta = Vector{Vector{Float64}}(0)
    for k = 1:length(theta_bar)
        # Get sampler around the non infinity elements of theta_bar[k]
        d = MvBall(n_finite, r, theta_bar[k][idx_finite])

        # Sample uniform point on the ball
        theta_new = [copy(theta_bar[k]) for _ in 1:n_sample_per_op]
        [theta_new[i][idx_finite] = rand(d) for i in 1:n_sample_per_op]

        # Add sample to list of points
        append!(theta, theta_new)
    end


    return theta

end

