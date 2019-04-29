#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=sched_mit_sloan_gpu
#SBATCH --time=0-24:00
#SBATCH -o /home/stellato/projects/mlopt/examples/output/output_gpu_%j.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bartolomeo.stellato@gmail.com

# Load cuda
module load sloan/cuda/9.0

# Activate environment
source activate python37

SLURM_PARTITION=`squeue -h -o "%P" -j$SLURM_JOB_ID`;
if [[ $SLURM_PARTITION == *"interactive"* ]]; then
    export IAI_LICENSE_FILE="/home/stellato/iai-licenses/${SLURMD_NODENAME}.lic"
elif [[ $SLURM_PARTITION == *"gpu"* ]]; then
    export IAI_LICENSE_FILE="/home/stellato/iai_gpu.lic"
    export GRB_LICENSE_FILE="/home/stellato/gurobi_gpu.lic"
fi

echo $IAI_LICENSE_FILE

# Include script
# python paper/benchmarks/transportation/transportation.py
# python paper/benchmarks/portfolio/portfolio.py
# python paper/benchmarks/facility/facility.py
# python paper/benchmarks/control/control.py



# Online
python online_optimization/control/online_control.py --horizon 40
# python online_optimization/control/online_control.py --horizon 30
# python online_optimization/control/online_control.py --horizon 20
# python online_optimization/control/online_control.py --horizon 10
