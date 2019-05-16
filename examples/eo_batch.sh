#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=5G
#SBATCH --partition=sched_mit_sloan_batch
#SBATCH --time=2-00:00
#SBATCH -o /pool001/stellato/output/output_%A_N%a.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bartolomeo.stellato@gmail.com

sh eo_script.sh
