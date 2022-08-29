#!/bin/bash

#SBATCH --gres=gpu:T4:1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mail-type=END,FAIL
#SBATCH --output=ncu_%j.out

set -euxo pipefail

# bytes metrics
BYTES_HBM=dram__bytes.sum  # device memory
BYTES_PCI_R=lts__t_sectors_aperture_sysmem_op_read.sum  # PCIe read
BYTES_PCI_W=lts__t_sectors_aperture_sysmem_op_write.sum # PCIe write
BYTES_L1=l1tex__t_bytes.sum
BYTES_L2=lts__t_bytes.sum

# FLOPS metrics
FLOPS_TENSOR=sm__inst_executed_pipe_tensor.sum
FLOPS_PREFIX=sm__sass_thread_inst_executed_op
FLOPS_SUFFIX=pred_on.sum

CC=`which gcc`
CXX=`which g++`

hostname -s

cd $WKDIR  # set your working dir here
pwd

echo "Using ncu to profile PINN-heat-equation"

echo "Group: time"
TIME_AVG=sm__cycles_elapsed.avg
TIME_RATE=sm__cycles_elapsed.avg.per_second
# NCU command usage: ncu --metrics $metric1,$metric2 --csv $application
srun ncu --metrics $TIME_AVG,$TIME_RATE --csv ./build/pinn >> time.csv

echo "Group: bytes_cache"
srun ncu --metrics $BYTES_L1,$BYTES_L2 --csv ./build/pinn >> bytes_cache.csv

echo "Group: bytes_hbm"
srun ncu --metrics $BYTES_HBM --csv ./build/pinn >> bytes_hbm.csv

echo "Group: bytes_pci"
srun ncu --metrics $BYTES_PCI_R,$BYTES_PCI_W --csv ./build/pinn >> bytes_pci.csv

echo "Group: tensor FLOPS"
FLOPS_TENSOR=sm__inst_executed_pipe_tensor.sum
srun ncu --metrics $FLOPS_TENSOR --csv ./build/pinn >> flops_tensor.csv

echo "Group: f16 FLOPS"
FLOPS_ADD="$FLOPS_PREFIX"_hadd_"$FLOPS_SUFFIX"
FLOPS_MUL="$FLOPS_PREFIX"_hmul_"$FLOPS_SUFFIX"
FLOPS_FMA="$FLOPS_PREFIX"_hfma_"$FLOPS_SUFFIX"
srun ncu --metrics $FLOPS_ADD,$FLOPS_MUL,$FLOPS_FMA --csv ./build/pinn >> flops_f16.csv

echo "Group: f32 FLOPS"
FLOPS_ADD="$FLOPS_PREFIX"_fadd_"$FLOPS_SUFFIX"
FLOPS_MUL="$FLOPS_PREFIX"_fmul_"$FLOPS_SUFFIX"
FLOPS_FMA="$FLOPS_PREFIX"_ffma_"$FLOPS_SUFFIX"
srun ncu --metrics $FLOPS_ADD,$FLOPS_MUL,$FLOPS_FMA --csv ./build/pinn >> flops_f32.csv

echo "Group: f64 FLOPS"
FLOPS_ADD="$FLOPS_PREFIX"_dadd_"$FLOPS_SUFFIX"
FLOPS_MUL="$FLOPS_PREFIX"_dmul_"$FLOPS_SUFFIX"
FLOPS_FMA="$FLOPS_PREFIX"_dfma_"$FLOPS_SUFFIX"
srun ncu --metrics $FLOPS_ADD,$FLOPS_MUL,$FLOPS_FMA --csv ./build/pinn >> flops_f64.csv
