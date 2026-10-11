#!/bin/bash
# Run a PIB experiment queue on Frontier, one worker per GCD.
#   sbatch -N <nodes> --export=ALL,QUEUE=<queue.jsonl> scripts/queue.sbatch
#SBATCH -A LRN036
#SBATCH -J pib-queue
#SBATCH -p batch
#SBATCH -t 02:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH -o /lustre/orion/csc662/proj-shared/xxiao/PIB/logs/%x-%j.out
set -euo pipefail
source /lustre/orion/csc662/proj-shared/xxiao/PIB/code/scripts/frontier_env.sh
cd "$PIB_ROOT/code"
DEADLINE=$(date -d "$(scontrol show job "$SLURM_JOB_ID" | grep -oP 'EndTime=\K\S+')" +%s)
echo "queue=$QUEUE nodes=$SLURM_NNODES deadline=$(date -d @"$DEADLINE")"
srun --gpu-bind=closest python scripts/queue_worker.py --queue "$QUEUE" --runs "$PIB_ROOT/runs" \
  --deadline "$DEADLINE" --reserve-minutes "${RESERVE_MINUTES:-20}"
