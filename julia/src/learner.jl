function tree(X::DataFrame,
              y::Vector{Int64};
              sparse=false,
              export_tree=false,
              problem::OptimizationProblem=nothing,
              output_folder::String="output")
    @printf "Learning Classification Tree\n"

    options = Dict(:max_depth => 10,
                   :minbucket => 1,
                   :cp => 0.001,
                  )
    if sparse
        options[:hyperplane_config] = [Dict(:sparsity => 2)]
        options[:fast_num_support_restarts] = 10
    end

    lnr = OT.OptimalTreeClassifier(;options...)
    OT.fit!(lnr, X, y)

    # Export tree
    if export_tree
        output_name = lowercase(split(string(typeof(problem)), ".")[end])
        date = string(Dates.format(Dates.now(), "yy-mm-dd_HH:MM:SS"))
        export_tree_name = joinpath(output_folder,
                                    output_name * "_" * date)
        println("Export tree to $(export_tree_name)")
        OT.writedot("$(export_tree_name).dot", lnr)
        run(`dot -Tpdf -o $(export_tree_name).pdf $(export_tree_name).dot`)
    end

    @printf "Learning completed.\n"

    return lnr

end

function predict(X::DataFrame,
                 lnr::OT.OptimalTreeClassifier,
                 enc2strategy::Vector{Strategy})

    # Predict active_constr
    y = OT.predict(lnr, X)

    # Convert predicted encodings to strategies
    return [enc2strategy[y[i]] for i in 1:length(y)]

end


"""
    predict_best_full(X, k, lnr, problem, enc2active_constr)

Predict best strategy from the best k ones from the tree learner returning
the full information including the problem solution, and the time.
"""
function predict_best_full(X::DataFrame,
                           k::Int64,                                 # k best values
                           lnr::OT.OptimalTreeClassifier,
                           problem::OptimizationProblem,
                           enc2strategy::Vector{Strategy})

    # Get number of points
	n_points = size(X, 1)

    # Predict probabilities for each point
    proba = OT.predict_proba(lnr, X)

	# Get data to return for each point
    strategy = Vector{Strategy}(n_points)
	x = Vector{Vector{Float64}}(n_points)
	y = Vector{Vector{Float64}}(n_points)
	time = Vector{Float64}(n_points)

    # Pick k best classes for each point
	for i = 1:n_points  # Iterate over all the points

        # Do we need to populate the problem data ?
        populate!(problem, X[i, :])

        # Get probabilities row
        p = Array(proba[i, :])'[:]

		# Get k largest ones
		classes = sortperm(p, rev=true)[1:k]

		# Decode all the classes
		strategy_classes = [enc2strategy[classes[j]] for j in 1:k]

		# Get x, y, time for each one of them and store best one
		x_temp = Vector{Vector{Float64}}(k)
		time_temp = Vector{Float64}(k)
		infeas_temp = Vector{Float64}(k)
        cost_temp = Vector{Float64}(k)

        # Iterate over all the classes
		for j = 1:k

            # Solve problem
            x_temp[j], time_temp[j] = solve(problem, strategy_classes[j])

			# Compute infeasibility
			infeas_temp[j] = infeasibility(x_temp[j], problem)

			# Compute cost function
            cost_temp[j] = cost(problem, x_temp[j])

		end

        idx_filter = find(infeas_temp .<= TOL)
        if length(idx_filter) > 0
            # Case 1: Feasible points
            # -> Get solution with minimum cost between
            # feasible ones
            idx_pick = idx_filter[indmin(cost_temp[idx_filter])]
        else
            # Case 2: No feasible points
            # -> Get solution with minimum infeasibility
            idx_pick = indmin(infeas_temp)
        end

        #  if i in 1:10
        #      @show infeas_temp
        #      @show cost_temp
        #      @show time_temp
        #      @show idx_pick
        #  end

		# Store value we are interested in
        x[i] = x_temp[idx_pick]
        time[i] = sum(time_temp)   # Sum all the solve times
        strategy[i] = strategy_classes[idx_pick]

    end

    # Return x, time, and strategy for all the points
    return x, time, strategy

end

"""
    predict_best(X, k, lnr, problem, enc2active_constr)

Predict best strategy from the best k ones from the tree learner.
"""
function predict_best(X::DataFrame,
                      k::Int64,                                 # k best values
                      lnr::OT.OptimalTreeClassifier,
                      problem::OptimizationProblem,
                      enc2strategy::Vector{Strategy})
    _, _, strategy = predict_best_full(X, k, lnr, enc2strategy)
    return strategy
end
