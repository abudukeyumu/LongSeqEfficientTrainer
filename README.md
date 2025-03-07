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
| per_device-mini_batch-epoch | Doc2Dial |  |  |  |  | quac |  |  |  |  | qrecc |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | Recall@1 | Recall@5 | Recall@20 | MRR@20 | NDCG@20 | Recall@1 | Recall@5 | Recall@20 | MRR@20 | NDCG@20 | Recall@1 | Recall@5 | Recall@20 | MRR@20 | NDCG@20 |
| bge-large-en-v1.5 | 32.55 | 66.13 | 86.98 | 47.57 | 56.08 | 47.93 | 74.26 | 92.02 | 59.81 | 67.22 | 32.76 | 74.27 | 95.63 | 50.83 | 61.28 |
| 32-16-epoch-1 | 48.29 | 81.47 | 94.11 | 62.78 | 70.01 | 60.39 | 89 | 98.23 | 72.6 | 78.72 | 67.31 | 95.7 | 99.61 | 79.59 | 84.53 |
| 32-16-epoch-1 | 48.72 | 82.05 | 94.11 | 63.33 | 70.7 | 58.29 | 88.16 | 98 | 71.07 | 77.49 | 66.24 | 95.63 | 99.71 | 78.98 | 84.09 |
| 32-16-epoch-1 | 48.54 | 82.03 | 94.16 | 63.22 | 70.56 | 57.98 | 88.58 | 98.04 | 71 | 77.46 | 66.27 | 95.77 | 99.75 | 78.95 | 84.07 |
