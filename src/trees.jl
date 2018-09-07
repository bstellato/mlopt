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
                 enc2active_constr::Vector{Vector{Int64}})

    # Predict active_constr
    y = OT.predict(lnr, X)

    # Convert encoding to actual active_constr
    active_constr = [enc2active_constr[y[i]] for i in 1:length(y)]

    return active_constr

end

function predict_best(X::DataFrame,
                      k::Int64,                                 # k best values
                      lnr::OT.OptimalTreeClassifier,
					  problem::OptimizationProblem,
                      enc2active_constr::Vector{Vector{Int64}})

    # Get number of points
	n_points = size(X, 1)

    # Predict probabilities for each point
    proba = OT.predict_proba(X)

	# Get vector of best active constraints for each point
	active_constr = Vector{Vector{Int64}}

    # Pick k best classes for each point
	classes = Vector{Vector{Int64}}(n_points)
	for i = 1:n_points  # Iterate over all the points
        # Get probabilities row
        p = Array(proba[i, :])'

		# Get best k ones
		classes[i] = sort(p, rev=true)[1:k]  # Pick k biggest ones

		# Decode all the classes
		active_constr_classes = [enc2active_constr[classes[i][j]] for j in 1:k]

		# Get x, y, time for each one of them
		# and store best one
		x = Vector{Vector{Float64}}(k)
		y = Vector{Vector{Float64}}(k)
		time = Vector{Vector{Float64}}(k)
		infeas = Vector{Float64}(k)
		cost = Vector{Float64}(k)

		for j = 1:k  # Iterate over all the classes
		    x[j], y[j], time[j] = solve(problem, active_constr_classes[j])

			# Compute infeasibility
			infeas[j] = infeasibility(x[j], problem)

			# Compute cost function
            cost[j] = cost(problem, x[j])

		end

        idx_filter = find(infeas .<= TOL)
        if any(idx_filter)
            # Case 1: Feasible points
            # -> Get solution with minimum cost between
            # feasible ones
            idx_pick = indmin(cost[idx_filter])
        else
            # Case 2: No feasible points
            # -> Get solution with minimum infeasibility
            idx_pick = indmin(infeas)
        end

		# Return stuff we are interested in
         x[idx_pick], y[idx_pick], time[idx_pick]

    end

    # TODO: Return x, y, active_constr for all the points
    return x, y, time, active_constr

end

