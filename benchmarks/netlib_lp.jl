# Code for running the tree learning on the Netlib LPs
include("../src/MyModule.jl")

lp_data_dir = joinpath("benchmarks", "lp_data")
files = [f for f in readdir(lp_data_dir) if contains(f, "mps")]



# For each file perform lerning
files = ["25fv47.mps"]  # TODO: remove this. Just to run only one file
# TODO: Preallocate variables for debugging
problem = []
theta_bar = []
for f in files
    # Extract problem data
    problem = MyModule.OptimizationProblem(joinpath(lp_data_dir, f))

    # Sample operation points
    theta_bar = MyModule.operation_points(problem, N=10)

    # Training: Sample from operation points within Balls

    # Testing: Sample from operation points within Balls

    # Evaluate performance (TODO: Fix infeasibility/suboptimality measures)


end


# Store output of all files
