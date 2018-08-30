using Base.Test
include("../src/MyModule.jl")
TOL = 1e-05

@testset "extract_active_constr" begin

    # Define LP
    srand(1)
    n = 20
    m = 30
    c = -randn(n)
    u = rand(m)
    l = rand(m) - 1
    A = sprandn(m, n, 0.8)
    problem = MyModule.OptimizationProblem(c, l, A, u)

    # Solve LP
    x, y, _ = MyModule.solve(problem)

    # Get active_constr
    active_constr = MyModule.active_constraints(problem)

    # Solve with active_constr
    x_active_constr, y_active_constr = MyModule.solve(problem, active_constr)

    @test norm(x - x_active_constr) <= TOL
    @test norm(y - y_active_constr) <= TOL

end

