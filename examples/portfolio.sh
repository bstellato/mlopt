#!/bin/zsh
#SBATCH -n 12
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu:volta:1
#SBATCH --partition=gpu
#SBATCH -o /home/gridsan/stellato/results/online/portfolio/portfolio_%A_N%a.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bartolomeo.stellato@gmail.com

# Activate environment
conda activate python37

# module load gurobi/8.0.1
export GRB_LICENSE_FILE="/home/software/gurobi/gurobi.lic"

# SLURM_PARTITION=`squeue -h -o "%P" -j$SLURM_JOB_ID`;
# if [[ $SLURM_PARTITION == *"interactive"* ]]; then
#     # TODO: Fix this
#     export IAI_LICENSE_FILE="/home/stellato/iai_interactive.lic"
# elif [[ $SLURM_PARTITION == *"gpu"* ]]; then
#     module load sloan/cuda/9.0
#     export IAI_LICENSE_FILE="/home/stellato/iai_gpu.lic"
# fi

# Run actual script
# python online_optimization/control/online_control.py --horizon $SLURM_ARRAY_TASK_ID
HDF5_USE_FILE_LOCKING=FALSE python online_optimization/portfolio/portfolio.py --sparsity $SLURM_ARRAY_TASK_ID

# Process data and put together with other results
python online_optimization/portfolio/process_data.py 
