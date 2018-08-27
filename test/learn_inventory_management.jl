using Base.Test
using OptimalTrees
OT = OptimalTrees
using Combinatorics
using ProgressMeter
include("../src/MyModule.jl")


# Learn Inventory control problem active sets mapping
# ----------------------------------------------------
#  c, l, A , u = gen_supply_chain_model(x0, w, T)


# Generate parameters (Initial state)
N_train = 500
T = 10            # Horizon
w = repmat([1.], T, 1)          # Constant demand
X_train = [randn(1) for i = 1:N_train]

# Get active_constr for each point
active_constr = Vector{Vector{Int64}}(N_train)
@showprogress 1 "Computing active constraints..." for i = 1:N_train
    c, l, A, u = MyModule.gen_supply_chain_model(X_train[i], w, T)
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
#  export_tree_name = string(Dates.format(Dates.now(), "yy-mm-dd_HH:MM:SS"))
#  println("Export tree to $(export_tree_name)")
#  OT.writedot("$(export_tree_name).dot", lnr)
#  run(`dot -Tpdf -o $(export_tree_name).pdf $(export_tree_name).dot`)


# Generate new dataset to predict
N_test = 100
X_test = [randn(1) for i = 1:N_test]

# Get active_constr for each point
active_constr_test = Vector{Vector{Int64}}(N_test)
@showprogress 1 "Computing active constraints..." for i = 1:N_test
    c, l, A, u = MyModule.gen_supply_chain_model(X_test[i], w, T)
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
@printf "Number of correct evaluations: %d\n" n_correct_eval

# Solve problems with learned active_constr
x_pred = Vector{Vector{Float64}}(length(y_pred))
x_test = Vector{Vector{Float64}}(length(y_pred))
for i = 1:length(y_pred)
    @show length(active_constr_pred[i])
    c, l, A, u = MyModule.gen_supply_chain_model(X_test[i], w, T)
    x_pred[i], _ = MyModule.solve_with_active_constr(c, l, A, u,
                                                     active_constr_pred[i])
    x_test[i], _ = MyModule.solve_lp(c, l, A, u)
end





