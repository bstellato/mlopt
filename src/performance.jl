# Functions to analyze the performance

function accuracy(active_constr_pred::Vector{Vector{Int64}},
                  active_constr_test::Vector{Vector{Int64}})
    n_total = length(active_constr_pred)
    n_correct = 0
    for i = 1:n_total
        if active_constr_pred[i] == active_constr_test[i]
            n_correct += 1
        else
            println("Bad prediction at index $i")
        end
    end

    return n_correct / n_total
end


function eval_performance(theta::Vector{Vector{Float64}},
                          lnr::OT.OptimalTreeClassifier,
                          problem::OptimizationProblem,
                          enc2active_constr::Vector{Vector{Int64}})

    # Get active_constr for each point
    active_constr_test = MyModule.active_constraints(theta, problem)

    # Predict active constraints
    active_constr_pred = MyModule.predict(theta, lnr, enc2active_constr)

    # Get statistics
    num_test = length(theta)
    num_train = lnr.prb_.data.features.n_samples
    n_theta = length(theta[1])
    n_active_sets = length(enc2active_constr)

    # accuracy
    test_accuracy = accuracy(active_constr_pred, active_constr_test)

    @show test_accuracy

    # TODO: Add radius?

    # These need an additional problem solution
    infeas = Vector{Float64}(num_test)
    subopt = Vector{Float64}(num_test)
    time_comp = Vector{Float64}(num_test)
    for i = 1:num_test
        populate!(problem, theta[i])
        x_ml, y_ml, time_ml = solve(problem, active_constr_pred[i])
        x_lp, y_lp, time_lp = solve(problem)

        # Compare time
        infeas[i] = infeasibility(x_ml, problem)
        subopt[i] = suboptimality(x_ml, x_lp, problem)
        time_comp[i] = (1 - time_ml/time_lp)*100
    end


    # Create dataframe and export it
    df = DataFrame(
                   problem = [string(typeof(problem))],
                   num_test = Int[num_test],
                   num_train = Int[num_train],
                   n_theta = Int[n_theta],
                   n_active_sets = Int[n_active_sets],
                   accuracy = [test_accuracy],
                   avg_infeas = [mean([k for k in infeas if k >= TOL])],
                   avg_subopt = [mean([k for k in subopt if k >= TOL])],
                   avg_time_improvement_perc = [mean(time_comp)],
                  )

    df_detail = DataFrame(
                          problem = repmat([string(typeof(problem))], num_test),
                          infeas = infeas,
                          subopt = subopt,
                          time_improvement_perc = time_comp
                         )

    return df, df_detail
end


function write_output(df::DataFrame,
                      df_detail::DataFrame,
                      problem::OptimizationProblem;
                     output_folder::String="output")
    output_name = joinpath(output_folder,
                           string(typeof(problem)))
    CSV.write("$(output_name).csv", df)
    CSV.write("$(output_name)_detail.csv", df_detail)
    nothing
end

function write_output(df::DataFrame,
                      df_detail::DataFrame;
                      file_name::String="results",
                     output_folder::String="output")
    output_name = joinpath(output_folder, file_name)
    CSV.write("$(output_name).csv", df)
    CSV.write("$(output_name)_detail.csv", df_detail)
    nothing
end

"""
    infeasibility(x_ml, problem)

Compute infeasibility weighted over the magnitude of l and u.
"""
function infeasibility(x::Vector{Float64},
                       problem::OptimizationProblem)
    l, A, u = problem.data.l, problem.data.A, problem.data.u

    normA = [norm(A[i, :], Inf) for i = 1:size(A, 1)]
    upper = max.(A * x - u, 0.) ./ max.(normA, abs.(u))
    lower = max.(l - A * x, 0.) ./ max.(normA, abs.(l))

    return norm(upper + lower)

end

"""
    suboptimality(x, x_opt, problem)

Compute suboptimality weighted over the magnitude of c.
"""
function suboptimality(x::Vector{Float64},
                       x_opt::Vector{Float64},
                       problem::OptimizationProblem)
    c = problem.data.c
    return (c' * x - c' * x_opt) / norm(c, Inf)
end

