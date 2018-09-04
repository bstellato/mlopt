# Code for running the tree learning on the Netlib LPs
using DataFrames
include("../src/MyModule.jl")

lp_data_dir = joinpath("benchmarks", "lp_data")
files = [f for f in readdir(lp_data_dir) if contains(f, "mps")]

# Sort files by size
file_sizes = [stat(f).size for f in files]
files = files[sortperm(file_sizes)]

# Take only first 5 files
#  files = files[3:5]

# For each file perform lerning
files = ["afiro.mps"]  # TODO: remove this. Just to run only one file

N_op = 10
N_train = 1000
N_test = 100

println("Data points")
println(" - Training: $(N_train)")
println(" - Testing: $(N_test)")
println(" - Operating points: $(N_op)")

radius_frac = 0.1

# TODO: Preallocate variables for debugging
problem = []
theta_bar = []
theta_train = []
theta_test = []
theta_finite = []
radius = 1.0
y_train = []

problem = MyModule.NetlibLP()

# Dataframes to store data
df = DataFrame()
df_detail = DataFrame()

for f in files

    # Update internal file name
    problem.file_name = joinpath(lp_data_dir, f)
    problem.to_read = true

    # Populate problem data using nominal model
    MyModule.populate!(problem)

    # Sample operation points
    theta_bar = MyModule.operation_points(problem, N=N_op)

    # Radius is 10% of the mean of the finite elements of theta_bar
    theta_finite = [t[.!Base.isinf.(t)] for t in theta_bar]
    radius = radius_frac * mean(norm.(theta_finite, 1))

    # Training: Sample from operation points within Balls
    theta_train = MyModule.sample(problem, theta_bar, radius, N=N_train)
    # Testing: Sample from operation points within Balls
    theta_test = MyModule.sample(problem, theta_bar, radius, N=N_test)

    # Train
    srand(1)

    # Get active_constr for each point
    y_train, enc2active_constr = MyModule.encode(MyModule.active_constraints(theta_train,
                                                                             problem))

    # Learn tree
    lnr = MyModule.tree(theta_train, y_train, export_tree=true, problem=problem)

    # Evaluate performance
    df_f, df_detail_f = MyModule.eval_performance(theta_test,
                                                  lnr, problem,
                                                  enc2active_constr)

    # Concatenate dataframes
    df = [df; df_f]
    df_detail = [df_detail; df_detail_f]

    # Write output to file
    MyModule.write_output(df, df_detail)

end


# Store output of all files
