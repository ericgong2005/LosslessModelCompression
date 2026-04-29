# LosslessModelCompression
Characterizing different algorithms for Lossless LLM Model Compression

## Usage
All code is executed on the Harvard Compute Cluster. To generate the synthetic model weights, use `sbatch Submit.slurm`. To run the evaluations on various pre-processing methods with the Huffman encoding algorithms and Zstd algorithms, use `sbatch BenchmarkHuffman.slurm` and `sbatch BenchmarkZstd.slurm` respectively. Generate plots of results by running `python Plot.py`