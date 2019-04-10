#!/bin/bash

# Include script
srun --pty \
    --ntasks=1 \
    --nodes=1 \
    --cpus-per-task=8 \
    --mem-per-cpu=4G \
    --gres=gpu:1 \
    --partition=sched_mit_sloan_gpu \
    --time=0-24:00 \
    zsh
    # -o /home/stellato/projects/mlopt/output/output_$(date '+%Y-%m-%dT%H:%M').txt \
