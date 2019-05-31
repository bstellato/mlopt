#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=sched_mit_sloan_gpu
#SBATCH --time=0-24:00
#SBATCH -o /pool001/stellato/output/output_%A_N%a.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bartolomeo.stellato@gmail.com

sh eo_script.sh
