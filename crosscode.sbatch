#!/bin/bash
#SBATCH --job-name=crosscode
#SBATCH --partition=soc-gpu-np
#SBATCH --qos=soc-gpu-np
#SBATCH --account=soc-gpu-np
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --time=10:00:00
#SBATCH --chdir=/uufs/chpc.utah.edu/common/home/u0337294/cs6966-project
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs

module --force purge
module load miniforge3/25.11.0

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /uufs/chpc.utah.edu/common/home/u0337294/conda_envs/cs6966-project
set -u

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/uufs/chpc.utah.edu/common/home/u0337294/conda_envs/cs6966-project/bin/python -u run_crosscode.py
