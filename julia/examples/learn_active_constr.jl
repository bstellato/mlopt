using Base.Test
using OptimalTrees
OT = OptimalTrees
using Combinatorics
using ProgressMeter
include("../src/MyModule.jl")

# Define LP
srand(1)
n_var = 2
n_constr_A = 10
n_constr = n_constr_A + n_var
A = [sprandn(n_constr_A, n_var, 0.5); eye(n_var)]
u_x = 5 * ones(n_var)
l_x = -5 * ones(n_var)

# Add upper and lower bounds in A and b
#  A = [A; eye(n); -eye(n)]
#  b = [b; ub; -lb]


# Define vector of c and b (sample points)
N = 500
c = Vector{Vector{Float64}}(N)
u = Vector{Vector{Float64}}(N)
l = Vector{Vector{Float64}}(N)
Θ = Vector{Vector{Float64}}(N)
for i = 1:N
    c[i] = .1 * randn(n_var)
    u[i] = 1. + .1 * rand(n_constr_A)
    l[i] = -1 -.1 * rand(n_constr_A)
    Θ[i] = [c[i]; u[i]; l[i]]
end

# Get active_constr for each point
active_constr = Vector{Vector{Int64}}(N)
@showprogress 1 "Computing active constraints..." for i = 1:N
    active_constr[i] = MyModule.get_active_constr(c[i], [l[i]; l_x], A, [u[i]; u_x])
end

# Get unique bases as numbers
y_train, unique_active_constr = MyModule.active_constr_to_number(active_constr)

# Convert data to matrix
X_train = vcat(Θ'...)

# Train tree
lnr = OT.OptimalTreeClassifier(max_depth = 20,
                               minbucket = 1,
                               cp = 0.000001)
OT.fit!(lnr, X_train, y_train)

# Export tree
#  export_tree_name = string(Dates.format(Dates.now(), "yy-mm-dd_HH:MM:SS"))
#  println("Export tree to $(export_tree_name)")
#  OT.writedot("$(export_tree_name).dot", lnr)
#  run(`dot -Tpdf -o $(export_tree_name).pdf $(export_tree_name).dot`)


# Generate new dataset to predict
N = 100
c_test = Vector{Vector{Float64}}(N)
u_test = Vector{Vector{Float64}}(N)
l_test = Vector{Vector{Float64}}(N)
Θ_test = Vector{Vector{Float64}}(N)
for i = 1:N
    c_test[i] = .1 * randn(n_var)
    u_test[i] = 1. + .1 * rand(n_constr_A)
    l_test[i] = -1. - .1 * rand(n_constr_A)
    Θ_test[i] = [c_test[i]; u_test[i]; l_test[i]]
end

# Get active_constr for each point
active_constr_test = Vector{Vector{Int64}}(N)
@showprogress 1 "Computing active constraints..." for i = 1:N
    active_constr_test[i] = MyModule.get_active_constr(c_test[i],
                                                       [l_test[i]; l_x], A,
                                                       [u_test[i]; u_x])
end

# Test Θ
X_test = vcat(Θ_test'...)

# Predict active_constr
y_pred = OT.predict(lnr, X_test)

# Convert encoding to actual active_constr
active_constr_pred = [unique_active_constr[y_pred[i]] for i in 1:length(y_pred)]

# Compare active_constr
n_correct_eval = 0
for i = 1:length(y_pred)
    if active_constr_pred[i] == active_constr_test[i]
        n_correct_eval += 1
    else
        println("Bad prediction at index $i")
    end
end


println("Results")
@printf "Number of correct evaluations: %d\n" n_correct_eval

# Solve problems with learned active_constr
x_pred = Vector{Vector{Float64}}(length(y_pred))
x_test = Vector{Vector{Float64}}(length(y_pred))
for i = 1:length(y_pred)
    x_pred[i], _ = MyModule.solve_with_active_constr(c_test[i],
                                                     [l_test[i]; l_x],
                                                     A,
                                                     [u_test[i]; u_x],
                                                     active_constr_pred[i])
    x_test[i], _ = MyModule.solve_lp(c_test[i],
                                     [l_test[i]; l_x],
                                     A,
                                     [u_test[i]; u_x])
end





