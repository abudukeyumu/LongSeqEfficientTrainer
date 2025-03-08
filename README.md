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
      <th rowspan="2">raw_train-batch_size</th>
      <th colspan="6" style="text-align: center;">Doc2Dial</th>
      <th colspan="6" style="text-align: center;">quac</th>
      <th colspan="6" style="text-align: center;">qrec</th>
    </tr>
    <tr>
      <th>Recall@1</th>
      <th>Recall@5</th>
      <th>Recall@20</th>
      <th>MRR@20</th>
      <th>NDCG@20</th>
      <th>avg</th>
      <th>Recall@1</th>
      <th>Recall@5</th>
      <th>Recall@20</th>
      <th>MRR@20</th>
      <th>NDCG@20</th>
      <th>avg</th>
      <th>Recall@1</th>
      <th>Recall@5</th>
      <th>Recall@20</th>
      <th>MRR@20</th>
      <th>NDCG@20</th>
      <th>avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>raw-4</td>
      <td>43.92</td>
      <td>75.73</td>
      <td>91.24</td>
      <td>58.00</td>
      <td>65.78</td>
      <td>66.93</td>
      <td>55.70</td>
      <td>88.02</td>
      <td>98.53</td>
      <td>69.40</td>
      <td>76.35</td>
      <td>77.60</td>
      <td>69.10</td>
      <td>95.63</td>
      <td>99.68</td>
      <td>80.65</td>
      <td>85.33</td>
      <td>86.48</td>
    </tr>
    <tr>
      <td>new-4</td>
      <td>43.44</td>
      <td>77.00</td>
      <td>91.98</td>
      <td>58.20</td>
      <td>66.14</td>
      <td>67.35</td>
      <td>55.87</td>
      <td>88.99</td>
      <td>98.30</td>
      <td>71.62</td>
      <td>77.99</td>
      <td>78.95</td>
      <td>68.60</td>
      <td>95.77</td>
      <td>99.68</td>
      <td>80.42</td>
      <td>85.16</td>
      <td>85.93</td>
    </tr>
    <tr>
      <td>raw-8</td>
      <td>47.27</td>
      <td>79.61</td>
      <td>93.37</td>
      <td>61.56</td>
      <td>69.06</td>
      <td>70.37</td>
      <td>60.02</td>
      <td>89.95</td>
      <td>98.65</td>
      <td>72.72</td>
      <td>78.92</td>
      <td>80.05</td>
      <td>66.95</td>
      <td>95.57</td>
      <td>99.61</td>
      <td>79.35</td>
      <td>84.33</td>
      <td>85.16</td>
    </tr>
    <tr>
      <td>new-8</td>
      <td>47.47</td>
      <td>80.22</td>
      <td>93.60</td>
      <td>61.86</td>
      <td>69.35</td>
      <td>70.70</td>
      <td>61.20</td>
      <td>90.17</td>
      <td>98.71</td>
      <td>73.49</td>
      <td>79.53</td>
      <td>80.62</td>
      <td>67.67</td>
      <td>95.41</td>
      <td>99.64</td>
      <td>79.70</td>
      <td>84.61</td>
      <td>85.41</td>
    </tr>
    <tr>
      <td>raw-32</td>
      <td>46.81</td>
      <td>81.04</td>
      <td>94.06</td>
      <td>61.81</td>
      <td>69.45</td>
      <td>70.63</td>
      <td>58.69</td>
      <td>88.20</td>
      <td>98.11</td>
      <td>71.27</td>
      <td>77.66</td>
      <td>78.79</td>
      <td>68.35</td>
      <td>96.34</td>
      <td>99.78</td>
      <td>80.34</td>
      <td>85.14</td>
      <td>86.39</td>
    </tr>
    <tr>
      <td>new-32</td>
      <td>48.13</td>
      <td>81.52</td>
      <td>94.11</td>
      <td>62.67</td>
      <td>70.13</td>
      <td>71.31</td>
      <td>60.61</td>
      <td>89.11</td>
      <td>98.21</td>
      <td>72.83</td>
      <td>78.89</td>
      <td>80.33</td>
      <td>67.38</td>
      <td>95.84</td>
      <td>99.71</td>
      <td>79.71</td>
      <td>84.65</td>
      <td>85.86</td>
    </tr>
  </tbody>
</table>
