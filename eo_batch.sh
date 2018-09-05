#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=sched_mit_sloan_batch
#SBATCH --time=0-24:00
#SBATCH -o /home/stellato/projects/ml_for_param_opt/output/output.txt
#SBATCH -e /home/stellato/projects/ml_for_param_opt/output/error.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bartolomeo.stellato@gmail.com

# Load Julia modules
module load julia/0.6.2
module load cplex/128

# Include script
julia -e 'include("benchmarks/netliblp.jl")'


