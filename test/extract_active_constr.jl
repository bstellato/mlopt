using Base.Test
include("../src/MyModule.jl")

@testset "extract_active_constr" begin

    # Define LP
    srand(1)
    n = 100
    m = 500
    c = randn(n)
    b = rand(m)
    A = sprandn(m, n, 0.5)


    # Solve LP
    x, y = MyModule.solve_lp(c, A, b)

    # Get active_constr
    active_constr = MyModule.get_active_constr(c, A, b)

    # Solve with active_constr
    x_active_constr, y_active_constr = MyModule.solve_with_active_constr(c, A, b, active_constr)

    @test norm(x - x_active_constr) <= TOL
    @test norm(y - y_active_constr) <= TOL

end

