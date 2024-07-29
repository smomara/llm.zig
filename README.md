# llm.zig

llm.zig is a Zig implemenation of a Large Language Model (LLM) trainer, inspired by Andrej Karpathy's llm.c project. The goal is to train LLMs entirely in Zig, aiming for performance that surpasses PyTorch.

## Features

* Pure Zig implementation of LLM training
* Customizable model parameters
* CPU-only (currently)

## Current Status

The project can currently train LLMs, but performance optimizations are ongoing. Future improvements will include:

* SIMD operations implementation (using Zig's `@Vector` builtin)
* Parallelization for improved performance
* CUDA kernel integration for GPU acceleration

## Getting Started

### Prerequisites

* Zig compiler
* Python (for running the prep script)

### Setup

1. Close the repository:
```
git clone https://github.com/yourusername/llm.zig.git
cd llm.zig
```
2. Run the prep script to initialize the tokenizer and tokenize the dataset:
```
python ./prep.py
```
3. Build the training executable:
```
zig build-exe train_gpt2.zig -O ReleaseFast
```
4. Run the training program:
```
./train_gpt2
```

## Customization

You can adjust various parameters in `train_gpt2.zig` to experiment with different model configurations.

> Note: keep the `vocab_size` and `padded_vocab_size` unchanged as they are determined by the tokenizer.
