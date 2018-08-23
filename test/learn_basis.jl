using Base.Test
using OptimalTrees
OT = OptimalTrees
using Combinatorics
using ProgressMeter
include("../src/MyModule.jl")

# Define LP
srand(1)
n = 2
m = 5
#  c = randn(n)
#  b = rand(m)
A = sprandn(m, n, 0.5)
ub = 5 * ones(n)
lb = -5 * ones(n)



# Add upper and lower bounds in A and b
A = [A; eye(n); -eye(n)]
#  b = [b; ub; -lb]


# Define vector of c and b (sample points)
N = 300
c = Vector{Vector{Float64}}(N)
b = Vector{Vector{Float64}}(N)
Θ = Vector{Vector{Float64}}(N)
for i = 1:N
    c[i] = 1 * randn(n)
    b[i] = 1 * rand(m)
    Θ[i] = [c[i]; b[i]]
end

# Get basis for each point
basis = Vector{Vector{Int64}}(N)
@showprogress 1 "Computing basis..." for i = 1:N
    basis[i] = MyModule.get_basis(c[i], A, [b[i]; ub; -lb])
end

# Get unique bases as numbers
y_train, unique_basis = MyModule.basis_to_number(basis)

# Convert data to matrix
X_train = vcat(Θ'...)

# Train tree
lnr = OT.OptimalTreeClassifier(max_depth = 5,
                               minbucket = 1,
                               cp = 0.0001)
OT.fit!(lnr, X_train, y_train)

# Export tree
export_tree_name = string(Dates.format(Dates.now(), "yy-mm-dd_HH:MM:SS"))
println("Export tree to $(export_tree_name)")
OT.writedot("$(export_tree_name).dot", lnr)
run(`dot -Tpdf -o $(export_tree_name).pdf $(export_tree_name).dot`)


# Generate new dataset to predict
N = 100
c_test = Vector{Vector{Float64}}(N)
b_test = Vector{Vector{Float64}}(N)
Θ_test = Vector{Vector{Float64}}(N)
for i = 1:N
    c_test[i] = 1 * randn(n)
    b_test[i] = 1 * rand(m)
    Θ_test[i] = [c_test[i]; b_test[i]]
end

# Get basis for each point
basis_test = Vector{Vector{Int64}}(N)
@showprogress 1 "Computing basis..." for i = 1:N
    basis_test[i] = MyModule.get_basis(c_test[i], A,
                                       [b_test[i]; ub; -lb])
end


# Test Θ
X_test = vcat(Θ_test'...)

# Predict basis
y_pred = OT.predict(lnr, X_test)

# Convert encoding to actual basis
basis_pred = [unique_basis[y_pred[i]] for i in 1:length(y_pred)]


# Compare basis
n_correct_eval = 0
for i = 1:length(y_pred)
    if basis_pred[i] == basis_test[i]
        n_correct_eval += 1
    end
end


println("Results")
@printf "Number of correct evaluations: %d\n" n_correct_eval









