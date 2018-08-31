# Code for running the tree learning on the Netlib LPs
include("../src/MyModule.jl")

lp_data_dir = joinpath("benchmarks", "lp_data")
files = [f for f in readdir(lp_data_dir) if contains(f, "mps")]



# For each file perform lerning
files = ["25fv47.mps"]  # TODO: remove this. Just to run only one file
problem = []
for f in files
    # Sample operation points
    problem = MyModule.OptimizationProblem(joinpath(lp_data_dir, f))

end

