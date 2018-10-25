#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=sched_mit_sloan_batch
#SBATCH --time=0-24:00
#SBATCH -o /home/stellato/projects/mlopt/output/output_$(date +%Y-%m-%d_%H-%M).txt
#SBATCH -e /home/stellato/projects/mlopt/output/error_$(date +%Y-%m-%d_%H-%M).txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bartolomeo.stellato@gmail.com

# Include script
$1


