populate!(::OptimizationProblem) = error("custom OptimizationProblem objects must define a `populate!` method")

"""
    @remove_unbounded_constraints A l u

Remove constraints where both l_i and u_i have infinite magnitude.
"""
macro remove_unbounded_constraints(A, l, u)

    return quote
        local _A = $(esc(A))
        local _l = $(esc(l))
        local _u = $(esc(u))
        local idx_constr = setdiff(1:length(_l), intersect(find(Base.isinf.(_u)), find(Base.isinf.(_l))))
        $(esc(A)) = _A[idx_constr, :]
        $(esc(l)) = _l[idx_constr]
        $(esc(u)) = _u[idx_constr]
    end
end

"""
    extract_problem_data(m)

Extract problem data from JuMP model
"""
function extract_problem_data(m::JuMP.Model)
    # Build internal model
    JuMP.build(m)

    # Extract data
    return extract_problem_data(m.internalModel)

end

function extract_problem_data(m::MathProgBase.AbstractLinearQuadraticModel)
    c = MathProgBase.getobj(m)
    A = [MathProgBase.getconstrmatrix(m); eye(length(c))]
    l = [MathProgBase.getconstrLB(m); MathProgBase.getvarLB(m)]
    u = [MathProgBase.getconstrUB(m); MathProgBase.getvarUB(m)]

    # Cap constraints with infinity
    [u[i] = Inf for i in 1:length(u) if u[i] >= 1e20]
    [l[i] = -Inf for i in 1:length(l) if l[i] <= -1e20]

    # Remove indices where both bounds are infinity
    MyModule.@remove_unbounded_constraints A l u

    return ProblemData(c, l, A, u)
end


function extract_problem_data(file_name::String)
    m = MathProgBase.LinearQuadraticModel(READ_SOLVER)
    MathProgBase.loadproblem!(m, file_name)
    return extract_problem_data(m)
end


#
# Create multivariate distribution sampler
#
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


