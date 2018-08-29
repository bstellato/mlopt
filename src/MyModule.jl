module MyModule

# Incldue packages
using OptimalTrees
using JuMP
using Plots
using MathProgBase
using Mosek
using CPLEX
using GLPKMathProgInterface


# Define constants
TOL = 1e-06
INFINITY = 1e15
SOLVER = CplexSolver(CPX_PARAM_SCRIND = 0)
#  SOLVER= GLPKSolverLP()
#  SOLVER= MosekSolver(QUIET=1)

include("active_constr.jl")  # Functions for solving and identifying active constraints
include("performance.jl")    # Functions for analyzing performance of the method

end

