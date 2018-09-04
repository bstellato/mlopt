mutable struct NetlibLP <: OptimizationProblem
    # Data in LP form
    data::ProblemData

    # Additional data to build the problem
    # (independent from the parameters)
    file_name::String

    # Empty constructor
    NetlibLP() = new()
end


function populate!(problem::NetlibLP)

    println("Called `lp_data`. Problem $(problem.file_name)")

    # Extract nominal problem data
    problem.data = extract_problem_data(problem.file_name)

end


function populate!(problem::NetlibLP, theta::Array{Float64})

    println("Called `lp_data`. Problem $(problem.file_name)")

    if !isdefined(problem, :data)
        # Extract nominal problem data if not defined
        problem.data = extract_problem_data(problem.file_name)
    end

    # Get problem dimensions
    n_var = length(problem.data.c)
    n_constr = length(problem.data.l)

    # Populate prblem data using theta
    problem.data.c = theta[1:n_var]
    problem.data.l = theta[n_var+1:n_var+n_constr]
    problem.data.u = theta[n_var + n_constr+1:end]

end


