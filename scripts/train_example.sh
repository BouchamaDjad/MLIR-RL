#!/bin/bash

# Define the resource requirements here using #SBATCH
#SBATCH -p compute
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH -c 28
#SBATCH --mem=64G
#SBATCH -t 07-00
#SBATCH -o /scratch/rb5953/MLIR-RL-2/logs/train_%J.out
#SBATCH -e /scratch/rb5953/MLIR-RL-2/logs/train_%J.err

# Resource requiremenmt commands end here

#Add the lines for running your code/application
module load miniconda-nobashrc
eval "$(conda shell.bash hook)"

# Activate any environments if required
conda activate main_env_5

# Set config file path
export CONFIG_FILE_PATH=config/example.json
# Execute the code
python /scratch/rb5953/MLIR-RL-2/train.py
