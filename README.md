# Efficient Long Sequence Contrastive Learning Framework

## Overview

This repository contains the implementation of a highly efficient training framework for long-sequence (up to 8192 tokens) contrastive learning tasks. By combining **extended positional embeddings** and **gradient cache (GradCache)** techniques, this framework optimizes memory usage and training efficiency, enabling large batch-size training on a single GPU.

### Key Features
- Support for **long-sequence input** (up to 8192 tokens)
- Efficient training with **large batch sizes** using gradient accumulation
- Customizable micro-batch training via `mini_batch_size` for fine-grained control
- Evaluation script for benchmarking model performance across multiple metrics

## File Structure

- **`origin_train.py`**: Original training script without large batch-size optimization
- **`efficient_train.py`**: Optimized training script with support for large batch sizes using gradient accumulation
- **`eval.sh`**: Evaluation script to compute performance metrics across multiple datasets and models

## Training

### Original Training
To run the original unoptimized training script:

python origin_train.py

### Efficient Training
To train using the optimized script with large batch size:

python efficient_train.py

#### Key Arguments
- ****`mini_batch_size`****: Defines the micro-batch size for each gradient accumulation step
- ****`per_device_train_batch_size-`****: Number of samples processed per device in each step
- ****`gradient_accumulation_steps-`****: Number of steps over which gradients are accumulated
Note: The effective batch size is calculated as:
effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
## Evaluation
To evaluate the trained model, run:
bash eval.sh
## Result
  The script computes the following metrics for three datasets on five different models:
Recall@1, Recall@5, Recall@20，MRR@20, NDCG@20
