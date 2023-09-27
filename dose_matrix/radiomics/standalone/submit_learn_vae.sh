#!/bin/bash

NGPUS=1
TIME="14:00:00"
VAE="N32_2"
BATCH_SIZE=32
ZOOM=10

COMMANDS_JOB="
module purge
module load cuda/11.7.0/gcc-11.2.0
source activate radiopreditool
set -x
which python
python learn_vae.py --vae $VAE --batch-size $BATCH_SIZE --zoom $ZOOM
"

sbatch \
    --job-name=vae_${VAE}_batch_${BATCH_SIZE}_zoom_$ZOOM \
    --output=./out/vae_${VAE}_batch_${BATCH_SIZE}_zoom_$ZOOM_${NGPUS}gpus --time=${TIME} \
    --ntasks=1 --partition=gpu --gres=gpu:${NGPUS} \
    --mem=100G --wrap="$COMMANDS_JOB"

