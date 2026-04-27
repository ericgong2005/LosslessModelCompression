#!/usr/bin/env bash
#SBATCH --job-name=GenerateWeights
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --time=12:00:00

module purge
module load python/3.9
python -m pip install --user numpy
python -m pip install --user zstandard

srun --cpu-bind=none python CompressionBenchmark.py
