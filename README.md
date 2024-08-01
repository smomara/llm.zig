# llm.zig

llm.zig is a Zig implemenation of a Large Language Model (LLM) trainer, inspired by Andrej Karpathy's llm.c project. The goal is to train LLMs primarily in Zig, aiming for performance that surpasses PyTorch.

## Features

* Primarily Zig implementation of LLM training
* Customizable model parameters
* CPU-optimized uisng OpenBLAS for matrix multiplication
* Outperforms Karpathy's train_gpt2.c and PyTorch on CPU

## Current Status

The project can currently train LLMs with impressive performance on CPU using OpenBLAS for efficient matrix operations. Future improvements will include:
* Training in comptime: Using Zig's powerful comptime features and build system to perform training during compilation.
* CUDA kernel integration for GPU acceleration

## Getting Started

### Prerequisites

* Zig compiler
* Python (for running the prep script)
* OpenBLAS

### Setup

1. Close the repository:
```bash
git clone https://github.com/somara/llm.zig.git
cd llm.zig
```
2. Run the prep script to initialize the tokenizer and tokenize the dataset:
```bash
python ./prep.py
```
3. Build and run the training executable:
```bash
# ReleaseFast gives by far the best performance
# but the build.zig will use Debug by default
zig build -Doptimize-ReleaseFast run
```

## Customization

You can adjust various parameters in `train_gpt2.zig` to experiment with different model configurations.

> Note: keep the `vocab_size` and `padded_vocab_size` unchanged as they are determined by the tokenizer.

## Performance

llm.zig now outperforms Karpathy's train_gpt.c and PyTorch on CPU. Will update soon to include benchmarks.

## License

Do literally anything you want with this code.

## Acknowledgements
* Andrej Karpathy for the original llm.c project
* The OpenBLAS prohject for their high performance BLAS implementation
