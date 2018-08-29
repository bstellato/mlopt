# Functions to analyze the performance


#  mutable struct PerformanceStatistics
#      num_train::Int64   # Number of training samples
#      num_test::Int64    # Number of testing samples
#      infeas::Float64  # Average infeasibility of the solution for incorrect
#      subopt::Float64  # Average suboptimality of the solution for incorrect
#      n_theta::Int64   # Dimension of the parameters
#      n_active_sets::Int64  # Number of relevant active sets from training
#      correct::Float64  # Percentage of correct predictions
#      r::Float64       # Sampling radius
#      time_comparison::Float64   # Percentage of time improvement
#  end

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
                          gen_problem::Function,
                          enc2active_constr::Vector{Vector{Int64}})

    # Get active_constr for each point
    active_constr_test = MyModule.active_constraints(theta_test, gen_function)

    # Predict active constraints
    active_constr_pred = MyModule.predict(theta_test, lnr, enc2active_constr)

    # Get statistics
    num_test = length(theta)
    num_train = lnr.prb_.data.features.n_samples
    n_theta = length(theta[1])
    n_active_sets = length(enc2active_constr)

    # accuracy
    test_accuracy = accuracy(active_constr_pred, active_constr_test)

    # TODO: Add radius?

    # TODO: Add fields that require solution of the problem

    # These need an additional problem solution
    # infeasibility
    # suboptimality
    # time comparison

    # Create dataframe and export it
    df = DataFrame(
                   problem = [String(Base.function_name(gen_problem))],
                   num_test = Int[num_test],
                   num_train = Int[num_train],
                   n_theta = Int[n_theta],
                   n_active_sets = Int[n_active_sets],
                   accuracy = Int[test_accuracy],
                   #  fn = Int[fn[x] for x in ecg_data],
                   #  score = [score[x] for x in ecg_data],
                  )
end



