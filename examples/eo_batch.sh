#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=6G
#SBATCH --partition=sched_mit_sloan_interactive
#SBATCH --time=2-00:00
#SBATCH -o /home/stellato/projects/mlopt/examples/output/output_batch_%j.txt
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
python online_optimization/control/online_control.py --horizon 40
# python online_optimization/control/online_control.py --horizon 30
# python online_optimization/control/online_control.py --horizon 20
# python online_optimization/control/online_control.py --horizon 10
