# ddKS - a d-dimensional Kolmogorov-Smirnov Test

*Alex Hagen<sup>1</sup>, Shane Jackson<sup>1</sup>, James Kahn<sup>2</sup>, Jan Strube<sup>1</sup>, Isabel Haide<sup>2</sup>, Karl Pazdernik<sup>1</sup>, and Connor Hainje<sup>1</sup>*

<sup>1</sup>: Pacific Northwest National Laboratory, 
<sup>2</sup>: Karlsruhe Institute of Technology

This code accompanies our paper submitted to IEEE Transactions on
Pattern Analysis and Machine Intelligence titled "Accelerated Computation of a
High Dimensional Kolmogorov-Smirnov Distance" ([arXiv](https://arxiv.org/abs/2106.13706)).  

As of 6/25/2021 there are 3 methods implemented:

* ddKS - d-dimensional KS test caclulated per
    * Variable splitting of space (all points, subsample, grid spacing)
* rdKS - ddKS approximation using distance from (d+1) corners
* vdKS - ddKS approximation calculating ddks distance between voxels instead of points



# Quickstart

Installation of `ddks` should be pretty easy, simple run

```bash
pip install git+https://github.com/pnnl/DDKS
```

or, if you want to develop on DDKS, simply clone this repository into a safe
spot on your computer and run

```bash
pip install -e .
```

from the top level of the repository.

Then, you can get started used the
repository by starting a `ddks` object and performing the distance calculation
on any pair of torch tensors that are `sample_size` x `dimension`.

```python
import torch
import ddks

p = torch.rand((100, 3))
t = torch.rand((50, 3))

calculation = ddks.methods.ddKS()
distance = calculation(p, t)
print(f"The ddKS distance is {distance}")
```

To operate on GPU, all you need to do is move the tensors to the device before
calculation:

```python
p = torch.rand((100, 3)).to('cuda:0')
t = torch.rand((50, 3)).to('cuda:0')

calculation = ddks.methods.ddKS()
distance = calculation(p, t)
```

If you want to use a different accelerated method, simply use
`ddks.methods.rdKS` or `ddks.methods.vdKS`. Note that rdKS and vdKS cannot use
GPU.

# Package Structure:
1. methods - Callable classes for xdks methods [x=d,r,v]
1. data - Contains several data generators to play around with
1. run_scripts - Contains an example run script 
1. Unit_tests - Contains unit tests for repo   
