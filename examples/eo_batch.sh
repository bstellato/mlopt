#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=sched_mit_sloan_interactive
#SBATCH --time=2-00:00
#SBATCH -o /home/stellato/projects/mlopt/output/output_batch_%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bartolomeo.stellato@gmail.com

# Activate environment
# source activate python36
source activate python37

SLURM_PARTITION=`squeue -h -o "%P" -j$SLURM_JOB_ID`;
if [[ $SLURM_PARTITION == *"interactive"* ]]; then
    export IAI_LICENSE_FILE="/home/stellato/iai_interactive.lic"
elif [[ $SLURM_PARTITION == *"gpu"* ]]; then
    export IAI_LICENSE_FILE="/home/stellato/iai_gpu.lic"
fi

# Include script
# python examples/paper/portfolio/portfolio.py
# python examples/paper/benchmarks/transportation/transportation.py
# python examples/online_optimization/control/online_control_condense.py
python online_optimization/control/online_control.py --horizon 20
