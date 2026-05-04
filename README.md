# LosslessModelCompression
Characterizing different algorithms for Lossless LLM Model Compression

## Usage
All code is executed on the Harvard Compute Cluster. To generate the synthetic model weights, use `sbatch Submit.slurm`. To run the evaluations on various pre-processing methods with the Huffman encoding algorithms and Zstd algorithms, use `sbatch BenchmarkHuffman.slurm` and `sbatch BenchmarkZstd.slurm` respectively. Generate plots of results by running `python Plot.py`

Please note that due to the large size of the Synthetic data, it is not pushed to this repository. Instead a sample version where 2e10 weights are generated, as opposed to 2e30, are included as an examplar under `RESULTS_2_10`.

## Directory Overview
`OldBenchmarks`, `OldPlots`, and `OldScripts` contain test trial run results, previous iterations of code to visualize results, and the generated visualizations respectively.

`PLOTS_UNIFIED` contains the most up-to-date visualizations of the final results, including the figures referenced in the paper.

`RESULTS_2_10`, as mentioned earlier, contains a scaled-down version of the synthetic data

`BenchmarkHuffman.slurm` and `sbatch BenchmarkZstd.slurm` are the scripts used to run the Compression benchmarking scripts `CompressionBenchmarkHuffman.py` and `CompressionBenchmark.py` respectively

`CMakeLists.txt` compiles the code to generate Synthetic data found in `GenerateWeights.cpp` and can be executed via `Submit.slurm`

`Plot.py` generates the results saved to `PLOTS_UNIFIED`