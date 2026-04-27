#!/usr/bin/env bash
#SBATCH --job-name=GenerateWeights
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --time=12:00:00

srun --cpu-bind=none python CompressionBenchmark.py
