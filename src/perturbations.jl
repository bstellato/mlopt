# Functions to perturb benchmark library LPs

"""
    operation_points(problem[, N=10])

Sample the operation points by perturbing the original problem data.
"""
function operation_points(problem::OptimizationProblem;
                          N::Int64=10)
    # Problem data
    c, l, u = problem.c, problem.l, problem.u
    n_var = length(c)
    n_constr = length(problem.l)

    # Perturb cost
    c_up = c + 0.1 * abs.(c)
    c_low = c - 0.1 * abs.(c)
    c_vec = [c_low + (c_up - c_low) .* rand(n_var) for _ in 1:N]

    # Perturb lower bound
    l_up = l
    l_low = l - 0.1 * abs.(l)
    delta = l_up - l_low
    [delta[i] = 0 for i = 1:n_constr if isinf(u[i])]   # Interval 0 if delta Inf
    l_vec = [l_low + delta .* rand(n_constr) for _ in 1:N]

    # Perturb upper bound
    u_up = u + 0.1 * abs.(u)
    u_low = u
    delta = u_up - u_low
    [delta[i] = 0 for i = 1:n_constr if isinf(u[i])]   # Interval 0 if delta Inf
    u_vec = [u_low + delta .* rand(n_constr) for _ in 1:N]

    # Stack points up
    return [[c_vec[i]; l_vec[i]; u_vec[i]] for i = 1:N]
end





# Create multivariate distribution sampler
struct MvBall <: Sampleable{Multivariate, Continuous}
    n::Int64  # Dimension
    r::Float64  # Radius
    #  d::  # Custom sampler
end
Base.length(s::MvBall) = s.n # return the length of each sample

function inc_gamma_lower_reg(a, x)
    return incGamma(a, x, false, true)
end



"""
Generate a single vector sample to x
The function initially samples the points using a Normal Distribution
with `randn`. Then the incomplete gamma function is used to map the
points radially to fit in the hypersphere of finite radius r with
a uniform spatial distribution.
"""
function Distributions._rand!{T<:Real}(s::MvBall, x::AbstractVector{T})
    n, r = s.n, s.r
    x_sample = randn(1, n)
    s2 = sum(x_sample.^2, 2)
    # NB. sf_gamma_inc_P(a, x) computes the normalized incomplete gamma function
    # for a and x. The arguments are swapped from the Matlab version
    fr = r*(sf_gamma_inc_P.(n/2, s2/2).^(1/n))./sqrt(s2)
    frtiled = repmat(fr, 1, n)
    x_sample = x_sample.*frtiled

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
    n_op = length(theta_bar)
    n_sample_per_op = floor(N / n_op)


    # Get values that are not infinity (from first element)
    idx_finite = find(.!isinf(theta_bar[1]))
    n_finite = length(idx_finite)

    # Get sampler
    d = MvBall(n_finite, r)

    # Get sampling points per operation point
    theta = []
    for k = 1:length(theta_bar)
        # Sample point and project it ont othe ball
        theta_temp = rand(d, n_sample_per_op)
        theta_temp *= (r / norm(theta_temp))

        theta_copy = copy(theta_bar[k])
        theta_copy[idx_finite] = theta_temp
        push!(theta, theta_copy)


    end



end

