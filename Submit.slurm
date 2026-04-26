#!/usr/bin/env bash
#SBATCH --job-name=GenerateWeights
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --time=12:00:00

srun -n 1 -c 48 --cpu-bind=none ./build/GenerateWeights