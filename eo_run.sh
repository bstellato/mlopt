#!/bin/bash
srun --pty --cpus-per-task=2 --mem=16G --constraint="centos7" --time=1-00:00 --partition=sched_mit_sloan_interactive julia -e 'include("benchmarks/netliblp.jl")'


