# Functions to analyze the performance

function accuracy(active_constr_pred::Vector{Vector{Int64}},
                  active_constr_test::Vector{Vector{Int64}})
    n_total = length(active_constr_pred)
    idx_correct = zeros(Int, n_total)
    for i = 1:n_total
        if active_constr_pred[i] == active_constr_test[i]
            idx_correct[i] = 1
        end
    end

    return sum(idx_correct) / n_total, idx_correct
end


function eval_performance(theta::DataFrame,
                          lnr::OT.OptimalTreeClassifier,
                          problem::OptimizationProblem,
                          enc2active_constr::Vector{Vector{Int64}};
                          k::Int64=1   # k best predicted values
                         )

    println("Performance evaluation")
    println("Compute active constraints over test set")

    # Get active_constr for each point
    x_test, y_test, time_test, active_constr_test = solve(theta, problem)

    # Get predicted active constraints for all the test points
    x_pred, y_pred, time_pred, active_constr_pred = predict_best_full(theta, k, lnr, problem, enc2active_constr)

    # Get statistics
    num_var = length(problem.data.c)
    num_constr = length(problem.data.l)
    num_test = size(theta, 1)
    num_train = lnr.prb_.data.features.n_samples
    n_theta = size(theta, 2)
    n_active_sets = length(enc2active_constr)

    # Compute infeasibility and suboptimality
    infeas = [infeasibility(x, problem) for x in x_pred]
    subopt = [suboptimality(x_pred[i], x_test[i], problem) for i = 1:num_test]
    time_comp = [(1 - time_pred[i]/time_test[i])*100 for i = 1:num_test]

    # accuracy
    test_accuracy, idx_correct = accuracy(active_constr_pred, active_constr_test)

    # Get problem name
    if isdefined(problem, :file_name)
        problem_name = lowercase(split(splitdir(problem.file_name)[end], ".")[end-1])
    else
        problem_name = lowercase(split(string(typeof(problem)), ".")[end])
    end

    # DEBUG
    @show sum(infeas .>= TOL)

    # Create dataframe and export it
    df = DataFrame(
                   problem = [problem_name],
                   radius = [problem.radius],
                   k = [k],
                   num_var = Int[num_var],
                   num_constr = Int[num_constr],
                   num_test = Int[num_test],
                   num_train = Int[num_train],
                   n_theta = Int[n_theta],
                   n_correct = Int[sum(idx_correct)],
                   n_active_sets = Int[n_active_sets],
                   accuracy = [test_accuracy],
                   n_infeas = [sum(infeas .>= TOL)],
                   avg_infeas = [mean(infeas)],
                   avg_subopt = [mean(subopt[find(infeas .<= TOL)])],  # Mean of feasible cases
                   max_infeas = [maximum(infeas)],
                   max_subopt = [maximum(subopt[find(infeas .<= TOL)])],
                   avg_time_improvement_perc = [mean(time_comp)],
                   max_time_improvement_perc = [maximum(time_comp)],
                  )

    df_detail = DataFrame(
                          problem = repmat([problem_name], num_test),
                          correct = idx_correct,
                          infeas = infeas,
                          subopt = subopt,
                          time_improvement_perc = time_comp
                         )

    return df, df_detail
end

"""
    infeasibility(x_ml, problem)

Compute infeasibility weighted over the magnitude of l and u.
"""
function infeasibility(x::Vector{Float64},
                       problem::OptimizationProblem)
    l, A, u = problem.data.l, problem.data.A, problem.data.u


    norm_A = [norm(A[i, :], Inf) for i = 1:size(A, 1)]

    upper = max.(A * x - u, 0.)
    lower = max.(l - A * x, 0.)

    # For the non zero ones, normalize
    [upper[i] /= max.(norm_A[i], abs.(u[i])) for i = 1:length(upper) if upper[i] >= TOL]
    [lower[i] /= max.(norm_A[i], abs.(l[i])) for i = 1:length(lower) if lower[i] >= TOL]

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

function write_output(df::DataFrame,
                      df_detail::DataFrame,
                      problem::OptimizationProblem;
                     output_folder::String="output")
    # Get problem name
    problem_name = lowercase(split(string(typeof(problem)), ".")[end])
    output_name = joinpath(output_folder,
                           problem_name)
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
