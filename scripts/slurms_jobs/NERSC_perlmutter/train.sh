#!/bin/bash

cat $0

# Environment
set -x

module load conda
conda activate pharma
export WORLD_SIZE=$SLURM_NTASKS
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export HF_DATASETS_CACHE=/pscratch/sd/d/dfarough/hf_cache # HuggingFace cache 
export HF_HOME=/pscratch/sd/d/dfarough/hf_cache

echo "▶ Starting job $SLURM_JOB_NAME - $SLURM_JOB_ID with the following script:"
echo "▶ Running training script..."

SCRIPT_PATH="/global/homes/d/dfarough/pff/pff/training/nersc_perlmutter_experiment.py"

srun python $SCRIPT_PATH $1