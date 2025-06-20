#!/bin/bash

#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 28
#SBATCH --qos=c2
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH -t 7-0:00:00
#SBATCH -o /scratch/db5394/job-logs/job-%j-ppo-train.out
#SBATCH -e /scratch/db5394/job-logs/job-%j-ppo-train.err


module purge

source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate "main-env-1"

# export llvm_build_path=/scratch/db5394/llvm-project/build
export CONFIG_FILE_PATH=/scratch/db5394/MLIR-RL-Fork/config/config.json

export PATH=$SCRATCH/gcc/bin:$SCRATCH/lld/bin:$PATH
export LD_LIBRARY_PATH=$SCRATCH/gcc/lib64:$LD_LIBRARY_PATH
export CXX=$SCRATCH/gcc/bin/g++
export LLVM_BUILD_PATH=/scratch/db5394/llvm-project/build

export PYTHONPATH="${LLVM_BUILD_PATH}/tools/mlir/python_packages/mlir_core/"

export JSON_BATCH_NUM=$2

# Print the day and time every n minutes in the background
N=15 # Set N to the desired number of minutes
(
    while true; do
        date '+%Y-%m-%d %H:%M:%S'
        sleep $((N * 60))
    done
) &

cd $SCRATCH/MLIR-RL-Fork/
python3 train.py
