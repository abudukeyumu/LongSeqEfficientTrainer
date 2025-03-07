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
<style type="text/css">
.tg  {border:none;border-collapse:collapse;border-spacing:0;}
.tg td{border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;
  padding:10px 5px;word-break:normal;}
.tg th{border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;font-weight:normal;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:center}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:center}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky" rowspan="2"></th>
    <th class="tg-c3ow" colspan="3">Average</th>
    <th class="tg-c3ow" colspan="3">Doc2Dial</th>
    <th class="tg-c3ow" colspan="3">QuAC</th>
    <th class="tg-c3ow" colspan="3">QReCC</th>
  </tr>
  <tr>
    <th class="tg-c3ow">top-1</th>
    <th class="tg-c3ow">top-5</th>
    <th class="tg-c3ow">top-20</th>
    <th class="tg-c3ow">top-1</th>
    <th class="tg-c3ow">top-5</th>
    <th class="tg-c3ow">top-20</th>
    <th class="tg-c3ow">top-1</th>
    <th class="tg-c3ow">top-5</th>
    <th class="tg-c3ow">top-20</th>
    <th class="tg-c3ow">top-1</th>
    <th class="tg-c3ow">top-5</th>
    <th class="tg-c3ow">top-20</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">bge-large-en-v1.5</td>
    <td class="tg-c3ow">37.75</td>
    <td class="tg-c3ow">71.55</td>
    <td class="tg-c3ow">91.54</td>		
    <td class="tg-c3ow">32.55</td>
    <td class="tg-c3ow">66.13</td>
    <td class="tg-c3ow">86.98</td>		
    <td class="tg-c3ow">47.93</td>
    <td class="tg-c3ow">74.26</td>
    <td class="tg-c3ow">92.02</td>		
    <td class="tg-c3ow">32.76</td>
    <td class="tg-c3ow">74.27</td>
    <td class="tg-c3ow">95.63</td>		
  </tr>
  <tr>		
    <td class="tg-0pky">bge-synthesisQA</td>
    <td class="tg-c3ow">51.72</td>
    <td class="tg-c3ow">83.12</td>
    <td class="tg-c3ow">95.02</td>		
    <td class="tg-c3ow">38.72</td>
    <td class="tg-c3ow">71.39</td>
    <td class="tg-c3ow">88.86</td>		
    <td 

