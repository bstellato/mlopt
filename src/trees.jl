function tree(X::Vector{Vector{Float64}},
              y::Vector{Int64};
              export_tree=false,
              problem::OptimizationProblem=nothing,
              output_folder::String="output")
    @printf "Learning Classification Tree\n"

    lnr = OT.OptimalTreeClassifier(max_depth = 10,
                                   minbucket = 1,
                                   # cp = 0.001
                                  )
    OT.fit!(lnr, vcat(X'...), y)

    # Export tree
    if export_tree
        output_name = string(typeof(problem))
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

function predict(X::Vector{Vector{Float64}},
                 lnr::OT.OptimalTreeClassifier,
                 enc2active_constr::Vector{Vector{Int64}})

    # Predict active_constr
    y = OT.predict(lnr, vcat(X'...))

    # Convert encoding to actual active_constr
    active_constr = [enc2active_constr[y[i]] for i in 1:length(y)]

    return active_constr

end

