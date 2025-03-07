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
<table>
  <thead>
    <tr>
      <th rowspan="2">per_device-mini_batch-epoch</th>
      <th colspan="5" style="text-align: center;">Doc2Dial</th>
      <th colspan="5" style="text-align: center;">quac</th>
      <th colspan="5" style="text-align: center;">qrec</th>
    </tr>
    <tr>
      <th>Recall@1</th>
      <th>Recall@5</th>
      <th>Recall@20</th>
      <th>MRR@20</th>
      <th>NDCG@20</th>
      <th>Recall@1</th>
      <th>Recall@5</th>
      <th>Recall@20</th>
      <th>MRR@20</th>
      <th>NDCG@20</th>
      <th>Recall@1</th>
      <th>Recall@5</th>
      <th>Recall@20</th>
      <th>MRR@20</th>
      <th>NDCG@20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>bge-large-en-v1.5</td>
      <td>32.55</td>
      <td>66.13</td>
      <td>86.98</td>
      <td>47.57</td>
      <td>56.68</td>
      <td>47.93</td>
      <td>74.26</td>
      <td>92.02</td>
      <td>59.81</td>
      <td>67.22</td>
      <td>32.76</td>
      <td>74.27</td>
      <td>95.63</td>
      <td>50.83</td>
      <td>61.28</td>
    </tr>
    <tr>
      <td>32-16-epoch-1</td>
      <td>48.29</td>
      <td>81.47</td>
      <td>94.11</td>
      <td>62.78</td>
      <td>70.01</td>
      <td>60.39</td>
      <td>89.00</td>
      <td>98.23</td>
      <td>72.60</td>
      <td>78.72</td>
      <td>67.31</td>
      <td>95.70</td>
      <td>99.61</td>
      <td>79.59</td>
      <td>84.53</td>
    </tr>
    <tr>
      <td>32-16-epoch-1</td>
      <td>48.72</td>
      <td>82.05</td>
      <td>94.16</td>
      <td>63.33</td>
      <td>70.70</td>
      <td>58.29</td>
      <td>88.16</td>
      <td>98.00</td>
      <td>71.07</td>
      <td>77.49</td>
      <td>66.24</td>
      <td>95.63</td>
      <td>99.71</td>
      <td>78.95</td>
      <td>84.09</td>
    </tr>
    <tr>
      <td>32-16-epoch-1</td>
      <td>48.54</td>
      <td>82.03</td>
      <td>94.16</td>
      <td>63.22</td>
      <td>70.56</td>
      <td>57.98</td>
      <td>88.58</td>
      <td>98.04</td>
      <td>71.00</td>
      <td>77.46</td>
      <td>66.27</td>
      <td>95.77</td>
      <td>99.75</td>
      <td>78.95</td>
      <td>84.07</td>
    </tr>
  </tbody>
</table>
