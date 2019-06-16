#!/bin/bash

# Activate environment
source activate python37

# module load gurobi/8.0.1
export GRB_LICENSE_FILE="/home/software/gurobi/gurobi.lic"

SLURM_PARTITION=`squeue -h -o "%P" -j$SLURM_JOB_ID`;
if [[ $SLURM_PARTITION == *"interactive"* ]]; then
    # TODO: Fix this
    export IAI_LICENSE_FILE="/home/stellato/iai_interactive.lic"
elif [[ $SLURM_PARTITION == *"gpu"* ]]; then
    module load sloan/cuda/9.0
    export IAI_LICENSE_FILE="/home/stellato/iai_gpu.lic"
fi

# Run actual script
# python online_optimization/control/online_control.py --horizon $SLURM_ARRAY_TASK_ID
python online_optimization/portfolio/portfolio.py --sparsity $SLURM_ARRAY_TASK_ID