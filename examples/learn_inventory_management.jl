using OptimalTrees
using ProgressMeter
using JuMP
include("../src/MyModule.jl")
OT = OptimalTrees


function gen_inventory_management_model(x0::Array{Float64},
                                     w::Array{Float64},
                                     T::Int64;
                                     M = 10.,  # Max ordering capacity
                                     K = 10.,  # Fixed ordering cost
                                     c = 3.,   # Variable ordering cost
                                     h = 10.,  # Storage cost
                                     p = 30.,  # Shortage cost
                                     bin_vars::Bool=false)
    # Define JuMP model
    m = Model(solver=MyModule.BUILD_SOLVER)

    # Variables
    @variable(m, x[i=1:T+1])
    @variable(m, u[i=1:T])
    @variable(m, y[i=1:T])  # Auxiliary: y[t] = max{h * x[t], -p * x[t]}
    (bin_vars) && (@variable(m, v[i=1:T], Bin))

    # Constraints
    @constraint(m, [i=1:length(x0)], x[i] == x0[i])
    @constraint(m, yh[t=1:T], y[t] >= h * x[t])
    @constraint(m, yp[t=1:T], y[t] >= -p * x[t])
    @constraint(m, evolution[t=1:T], x[t + 1] == x[t] + u[t] - w[t])
    @constraint(m, [t=1:T], u[t] >= 0)
    if bin_vars
        @constraint(m, [t=1:T], u[t] <= M * v[t])
    end

    # Cost
    if bin_vars
        @objective(m, Min, sum(y[i] + c * u[i] + K * v[i]  for i in 1:T))
    else
        @objective(m, Min, sum(y[i] + c * u[i] for i in 1:T))
    end

    # Extract problem data
    return MyModule.extract_problem_data(m)

end

# Learn Inventory management problem active sets mapping
# ----------------------------------------------------
srand(1)

# Generate parameters (Initial state)
N_train = 500
T = 50            # Horizon
w = repmat([1.3], T, 1)          # Constant demand
X_train = [randn(1) for i = 1:N_train]

# Get active_constr for each point
active_constr = Vector{Vector{Int64}}(N_train)
@showprogress 1 "Computing active constraints..." for i = 1:N_train
    c, l, A, u = gen_inventory_management_model(X_train[i], w, T)
    active_constr[i] = MyModule.get_active_constr(c, l, A, u)
end

# Get unique bases as numbers
y_train, unique_active_constr = MyModule.active_constr_to_number(active_constr)

# Train tree
lnr = OT.OptimalTreeClassifier(max_depth = 20,
                               minbucket = 1,
                               cp = 0.000001)
OT.fit!(lnr, vcat(X_train'...), y_train)

# Export tree
export_tree_name = string(Dates.format(Dates.now(), "yy-mm-dd_HH:MM:SS"))
println("Export tree to $(export_tree_name)")
OT.writedot("$(export_tree_name).dot", lnr)
run(`dot -Tpdf -o $(export_tree_name).pdf $(export_tree_name).dot`)


# Generate new dataset to predict
N_test = 100
X_test = [randn(1) for i = 1:N_test]

# Get active_constr for each point
active_constr_test = Vector{Vector{Int64}}(N_test)
@showprogress 1 "Computing active constraints..." for i = 1:N_test
    c, l, A, u = gen_inventory_management_model(X_test[i], w, T)
    active_constr_test[i] = MyModule.get_active_constr(c, l, A, u)
end

# Predict active_constr
y_pred = OT.predict(lnr, vcat(X_test'...))

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
@printf "Number of correct evaluations: %d/%d\n" n_correct_eval N_test

# Solve problems with learned active_constr
x_pred = Vector{Vector{Float64}}(N_test)
x_test = Vector{Vector{Float64}}(N_test)
time_ml = Vector{Float64}(N_test)
time_lp = Vector{Float64}(N_test)
for i = 1:length(y_pred)
    c, l, A, u = gen_inventory_management_model(X_test[i], w, T)
    time_ml[i] = @elapsed x_pred[i], _ = MyModule.solve_with_active_constr(c, l, A, u,
                                                     active_constr_pred[i])
    time_lp[i] = @elapsed x_test[i], _ = MyModule.solve_lp(c, l, A, u)

    if abs(c'* x_pred[i] - c' * x_test[i]) > 1e-05
        println("Not matching cost at problem $i. Difference $(c'* x_pred[i] - c' * x_test[i])")
    end
end

@printf "Time LP = %.2e\n" mean(time_lp)
@printf "Time ML = %.2e\n" mean(time_ml)





