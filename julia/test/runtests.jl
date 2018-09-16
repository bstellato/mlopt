@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

# Define global variables
TOL = 1e-05

# Start tests timer
tic()

# Test classes
@time @testset "extract_basis.jl" begin include("extract_basis.jl") end

# Examples from test/examples/ folder
#  @testset "examples" begin
#    @time @testset "example_1" begin include("examples/example_1.jl") end
#  end

# End tests timer
toc()
