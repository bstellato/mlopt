#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=20G
#SBATCH --partition=sched_mit_sloan_interactive
#SBATCH --time=4-00:00
#SBATCH -o /pool001/stellato/output/output_%A_N%a.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bartolomeo.stellato@gmail.com

# Activate environment
# source activate python36
source activate python37

# module load gurobi/8.0.1
export GRB_LICENSE_FILE="/home/software/gurobi/gurobi.lic"
# export GRB_LICENSE_FILE="/home/stellato/gurobi_interactive.lic"

SLURM_PARTITION=`squeue -h -o "%P" -j$SLURM_JOB_ID`;
if [[ $SLURM_PARTITION == *"interactive"* ]]; then
    export IAI_LICENSE_FILE="/home/stellato/iai_interactive.lic"
elif [[ $SLURM_PARTITION == *"gpu"* ]]; then
    export IAI_LICENSE_FILE="/home/stellato/iai_gpu.lic"
fi

# Include script
python online_optimization/control/online_control.py --horizon $SLURM_ARRAY_TASK_ID
