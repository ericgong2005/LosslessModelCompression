#!/usr/bin/env bash
#SBATCH --job-name=GenerateWeights
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --time=12:00:00

which python
python --version
python -m pip --version

python -m pip install numpy zstandard

python -c "import sys; print(sys.executable)"
python -c "import numpy; print(numpy.__version__)"
python -c "import zstandard; print(zstandard.__version__)"

srun --cpu-bind=none python CompressionBenchmark.py
