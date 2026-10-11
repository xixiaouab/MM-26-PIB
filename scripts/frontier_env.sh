#!/bin/bash
# OLCF Frontier: ROCm 7.1 + PyTorch 2.10 environment for PIB jobs.
module purge
module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load rocm/7.1.1
module load craype-accel-amd-gfx90a

export PIB_ROOT=${PIB_ROOT:-/lustre/orion/csc662/proj-shared/xxiao/PIB}
export PATH=$PIB_ROOT/envs/pib/bin:$PATH
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}
export PYTHONNOUSERSITE=1
export PYTHONPATH=$PIB_ROOT/code:${PYTHONPATH:-}
export HF_HOME=$PIB_ROOT/hf_home
export HF_HUB_OFFLINE=1
export MIOPEN_USER_DB_PATH=/tmp/miopen-db-${SLURM_JOB_ID:-manual}-${SLURM_PROCID:-0}
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen-cache-${SLURM_JOB_ID:-manual}-${SLURM_PROCID:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
