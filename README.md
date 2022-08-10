# Solving 2D Heat Equation with PINN
This project is to solve a 2D heat equation with [PINN](https://en.wikipedia.org/wiki/Physics-informed_neural_networks).
 The math description and Python implementation is given by the [Jupyter script](./PhasmExampleHeatEquation.ipynb).
 It is one of the JLab EPSCI [PHASM](https://github.com/nathanwbrei/phasm) examples.

This stand-alone repo is created to test the libtorch C++ APIs without considering
 the compatibility to the other PHASM codebase.
 It can also be used as the backup of the main repo.

## Envirenment
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
 (looks like `glibc 2.27` is required) problem.
 Then I use the `pre-cxx11 ABI libtorch` for now.


## References
- [PINN original paper](https://www.sciencedirect.com/science/article/pii/S0021999118307125?casa_token=3bln19-QiY8AAAAA:fljJ0paZDeCUJFpWkSxJQrd1xGDEnrUxdXOIWfpZZ7N0MnevxvVNLDEEEMyzX2_IRkX7Hco9YME): Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations
- [Pytorch C++ API](https://pytorch.org/cppdocs/)
- PINN Pytorch implementation: https://github.com/jayroxis/PINNs