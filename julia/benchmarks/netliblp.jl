# Code for running the tree learning on the Netlib LPs
using DataFrames
include("../src/MyModule.jl")

# Repeatability
srand(1)

lp_data_dir = joinpath("benchmarks", "lp_data")
files = [f for f in readdir(lp_data_dir) if contains(f, "mps")]

# Sort files by size
file_sizes = [stat(f).size for f in files]
files = files[sortperm(file_sizes)]

# Take only first 5 files
#  files = [files[1]]   # Slow
#  files = [files[2]]   #  Very slow
#  files = [files[3]]   # ADLITTLE
#  files = [files[4]]   # AFIRO
#  files = [files[5]]   # AGG
#  files = [files[6]]   # AGG2
#  files = [files[7]]   # AGG3  Up to now r = 0.01
#  files = [files[8]]   # BANDM
#  files = [files[9]]   # BEACONFD
#  files = [files[10]]  # BLEND
#  files = [files[11]]  # BNL1
#  files = [files[12]]  # BNL2
files = files[3:end]



n_op = 10
n_train = 1000
n_test = 100
n_modes = 20

radius_vec = logspace(-7., 1.5, 30)

println("Data points")
println(repeat("-", 60))
println(" - Training: $(n_train)")
println(" - Testing: $(n_test)")
println(" - Operating points: $(n_op)")
println(" - Ideal different active constraints: $(n_modes)")

# TODO: Remove preallocated variables for debugging
problem = []
theta_bar = []
theta_train = []
theta_test = []
theta_finite = []
y_train = []

problem = MyModule.NetlibLP()

# Dataframes to store data
df = DataFrame()
df_detail = DataFrame()

for f in files
    println("File name: $(f)")
    println(repeat("-", 60))

    # Update internal file name
    problem.file_name = joinpath(lp_data_dir, f)
    problem.to_read = true

    # Populate problem data using nominal model
    MyModule.populate!(problem)

    # Print some details
    println(" - Variables: $(length(problem.data.c))")
    println(" - Constraints: $(length(problem.data.l))")
    println(" - Nonzeros: $(nnz(problem.data.A))")

    # Sample operation points
    theta_bar = MyModule.operation_points(problem, N=n_op)

    # Predefine variable for visibility outside the for loop
    enc2active_constr = Vector{Vector{Int64}}(0)
    println("Finding best perturbation radius")
    for r in radius_vec

        println("Radius $(r)")
        problem.radius = r

        # Training: Sample from operation points within Balls
        theta_train = MyModule.sample(problem, theta_bar, N=n_train)

        # Get active_constr for each point
        y_train, enc2active_constr = MyModule.encode(MyModule.active_constraints(theta_train,
                                                                                 problem))

        println("Found $(length(enc2active_constr)) active constraints")
        if length(enc2active_constr) > n_modes
            println("Active constraints greater than $(n_modes). Stop with radius $(r)")
            break
        end

        if r == radius_vec[end]
            println("Could not find more than $(n_modes) active constraints. Stop with radius $(r)")
        end

    end

    # Testing: Sample from operation points within Balls
    theta_test = MyModule.sample(problem, theta_bar, N=n_test)

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
    MyModule.write_output(df, df_detail, file_name="results_netliblp")

end


# Store output of all files
