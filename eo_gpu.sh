#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --partition=sched_mit_sloan_gpu
#SBATCH --time=0-24:00
#SBATCH -o /home/stellato/projects/mlopt/output/output_gpu_%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bartolomeo.stellato@gmail.com

# Include script
python examples/paper/portfolio/portfolio_cont.py
