module MyModule

# Incldue packages
using ProgressMeter
using OptimalTrees
OT = OptimalTrees
using JuMP
using Plots
using MathProgBase
using CPLEX
using OSQP


# Define constants
TOL = 1e-06
INFINITY = 1e15
SOLVER = CplexSolver(CPX_PARAM_SCRIND = 0)
#  SOLVER= GLPKSolverLP()
#  SOLVER= MosekSolver(QUIET=1)
BUILD_SOLVER = OSQPMathProgBaseInterface.OSQPSolver(verbose=false)

include("types.jl")  # Functions for solving and identifying active constraints
include("active_constr.jl")  # Functions for solving and identifying active constraints
include("trees.jl")    # Learn and predict using Optimal Trees
include("performance.jl")    # Functions for analyzing performance of the method
include("utils.jl")    # Functions for analyzing performance of the method

end

