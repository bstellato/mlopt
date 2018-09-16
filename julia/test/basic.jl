using Base.Test
include("../src/MyModule.jl")

#  @testset "basic" begin

T = 10
x0 = 3
w = 3 * rand(T)  # Disturbance
MyModule.solve_supply_chain(x0, w, T)


lnr = MyModule.estimate_cost(w, T)

#  end

