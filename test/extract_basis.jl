using Base.Test
include("../src/MyModule.jl")

@testset "extract_basis" begin

    # Define LP
    srand(1)
    n = 100
    m = 500
    c = randn(n)
    b = rand(m)
    A = sprandn(m, n, 0.5)


    # Solve LP
    x, y = MyModule.solve_lp(c, A, b)

    # Get basis
    basis = MyModule.get_basis(c, A, b)

    # Solve with basis
    x_basis, y_basis = MyModule.solve_with_basis(c, A, b, basis)

    @test norm(x - x_basis) <= TOL
    @test norm(y - y_basis) <= TOL

end

