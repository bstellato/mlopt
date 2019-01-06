#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=sched_mit_sloan_interactive
#SBATCH --time=0-24:00
#SBATCH -o /home/stellato/projects/mlopt/output/output_%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bartolomeo.stellato@gmail.com

# Activate environment
source activate python36

# Include script
# python examples/paper/portfolio/portfolio.py
python examples/paper/benchmarks/transportation/transportation.py
