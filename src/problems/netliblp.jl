mutable struct NetlibLP <: OptimizationProblem
    # Data in LP form
    data::ProblemData

    # Additional data to build the problem
    # (independent from the parameters)
    file_name::String
    to_read::Bool

    # Empty constructor
    NetlibLP() = new()
end


function populate!(problem::NetlibLP)

    if problem.to_read
        # Extract nominal problem data
        problem.data = extract_problem_data(problem.file_name)
        problem.to_read = false
    end

end


function populate!(problem::NetlibLP, theta::Array{Float64})

    if problem.to_read
        # Extract nominal problem data if not defined
        problem.data = extract_problem_data(problem.file_name)
        problem.to_read = false
    end

    # Get problem dimensions
    n_var = length(problem.data.c)
    n_constr = length(problem.data.l)

    # Populate prblem data using theta
    problem.data.c = theta[1:n_var]
    problem.data.l = theta[n_var+1:n_var+n_constr]
    problem.data.u = theta[n_var + n_constr+1:end]

end


