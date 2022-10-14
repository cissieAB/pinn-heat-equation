# Solving 2D Heat Equation with PINN
This project is to solve a 2D heat equation with [PINN](https://en.wikipedia.org/wiki/Physics-informed_neural_networks).
 The math description and Python implementation is given by the [Jupyter script](https://github.com/nathanwbrei/phasm/blob/main/python/PhasmExampleHeatEquation.ipynb).
 It is one of the JLab EPSCI [PHASM](https://github.com/nathanwbrei/phasm) examples.

This stand-alone repo is created to test the libtorch C++ APIs without
 considering the compatibility to the other PHASM codebase.
 It can also be used as the backup of the main repo.

## Environment
The repo is tested on the JLab ifarm GPU node with below configurations.

```
glibc(ldd): 2.17
cmake: 3.21.1
gcc: 10.2.0
CUDA: 11.4
GPU: one Tesla T4 or one TitanRTX
cuDNN: 8.4.1
libtorch: 1.12.1 + Linux + LibTorch + C++/Java + cu11.3 + pre-cxx11abi
OS: CentOS Linux 7 (Core)
Kernel: Linux 3.10.0-1160.71.1.el7.x86_64
```

As I have no root control of the system, the library version problem is not well handled.
 `cxx11 ABI libtorch` should be used but failed with a low `glibc`
 (description [here](https://github.com/nathanwbrei/phasm/blob/gpu/farm_guide.md#notes)) problem.
 Then I use the `pre-cxx11 ABI libtorch` for now.

## Build the project

```bash
# add torch lib to path
export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH

# build project
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="/path/to/libtorch;/path/to/cuDNN" ..
make
```

### `libtorch` and `cuDNN` installation
[`cuDNN`](https://developer.nvidia.com/rdp/cudnn-download) and
 [`libtorch`](https://pytorch.org/get-started/locally/)
 can be accessed at the official sites.
 The packages I am using are given as below.

```bash
cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive.tar.xz
libtorch-cxx11-abi-shared-with-deps-1.12.1+cu113.zip  # cxx11 ABI, not used
libtorch-shared-with-deps-1.12.0+cu113.zip  # Pre-cxx11 ABI, used in this repo
```

Unzip these packages, and they will make up of your `/path/to/libtorch` and `/path/to/cuDNN`.

## NCU profiling
A [ncu profiling](docs/prof_res.md) is conducted on the training process to study the behaviours
 of the kernels. The [Roofline model](https://crd.lbl.gov/divisions/amcr/computer-science-amcr/par/research/roofline/)
 is utilized to identify the bottleneck of the application.
 
## TODOs
- [ ] Add the original Gauss-Seidel iteration and timing functions for the whole process. 
- [ ] Reshape the NN structure to achieve higher throuput. 

## References
- [PINN original paper](https://www.sciencedirect.com/science/article/pii/S0021999118307125?casa_token=3bln19-QiY8AAAAA:fljJ0paZDeCUJFpWkSxJQrd1xGDEnrUxdXOIWfpZZ7N0MnevxvVNLDEEEMyzX2_IRkX7Hco9YME): Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations
- [Pytorch C++ API documenation](https://pytorch.org/cppdocs/)
- [Pytorch C++ examples](https://github.com/pytorch/examples/tree/main/cpp)
- PINN Pytorch implementation: https://github.com/jayroxis/PINNs
- Roofline profiling paper: https://arxiv.org/abs/2009.02449
