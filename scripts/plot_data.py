import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('ggplot')

MEM_ROOFS = {'HBM': 241.6, 'L2': 5760.9, 'L1': 4051.9}  # measured memory BW in GB/s
CMP_ROOFS = {'FP32': 8100}  # measured F32 performance in GFLOP/s

NX = 200
X_MIN = -2.5
X_MAX = 2.0
Y_MIN = 0.01
Y_MAX = 10000


def plot_time(csv_path):
    df = pd.read_csv(csv_path)
    time_us, kernel_id = df['time_us'], df['kernel_id']

    fig1, ax1 = plt.subplots()
    ax1.pie(time_us, labels=kernel_id, autopct='%2.1f%%')
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.title('Time distribution of the kernels')
    plt.savefig('time_distribution.png')


def get_ridge_data():
    y_L1, y_L2, y_HBM = [], [], []
    x = np.logspace(X_MIN, X_MAX, NX)

    def helper(target, val_cap, val_bw, val_x):
        target.append(val_cap) if val_x * val_bw >= val_cap else target.append(val_x * val_bw)

    for i in range(NX):
        helper(y_L1, CMP_ROOFS['FP32'], MEM_ROOFS['L1'], x[i])
        helper(y_L2, CMP_ROOFS['FP32'], MEM_ROOFS['L2'], x[i])
        helper(y_HBM, CMP_ROOFS['FP32'], MEM_ROOFS['HBM'], x[i])

    return x, y_L1, y_L2, y_HBM


def plot_roofline(csv_path):
    df = pd.read_csv(csv_path)
    L1_AI, L2_AI, HBM_AI, f32_GFLOPS, time_us = \
        df['L1_AI'], df['L2_AI'], df['HBM_AI'], df['f32_GFLOPS'], \
        df['time_us']/8  # df['time_us']/8 to adjust maker size

    x, y_L1, y_L2, y_HBM = get_ridge_data()  # the roofline ridges

    fig = plt.figure(1)
    plt.clf()
    ax = fig.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Arithmetic Intensity [FLOPs/Byte]')
    ax.set_ylabel('Performance [GFLOP/sec]')

    ax.set_xlim(10 ** X_MIN, 10 ** X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)

    # plot the ridges
    ax.plot(x, y_L1, c='g', label='L1 limit')
    ax.plot(x, y_L2, c='r', label='L2 limit')
    ax.plot(x, y_HBM, c='b', label='HBM limit')

    # plot the AI dots
    ax.scatter(L1_AI, f32_GFLOPS, c='green', label='L1 AI',
                   alpha=0.5, edgecolors='face', marker='o', s=time_us)  # s=time_us, to control maker size
    ax.scatter(L2_AI, f32_GFLOPS, c='red', label='L2 AI',
               alpha=0.5, edgecolors='face', marker="^", s=time_us)
    ax.scatter(HBM_AI, f32_GFLOPS, c='blue', label='HBM AI',
               alpha=0.5, edgecolors='face', marker="s", s=time_us)
    plt.legend()

    plt.title('Hierarchical roofline on a T4 GPU')
    plt.savefig('roofline.png')


plot_time('../data/time_distribution.csv')
plot_roofline('../data/roofline_scatter.csv')
