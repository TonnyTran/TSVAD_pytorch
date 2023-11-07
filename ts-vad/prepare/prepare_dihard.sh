#!/bin/bash
#SBATCH --partition=SCSEGPU_M1 
#SBATCH --qos=q_amsai 
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1 
#SBATCH --mem=1G 
#SBATCH --job-name=preparedihard
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

python /home/msai/adnan002/repos/TSVAD_pytorch/ts-vad/prepare/prepare_dihard.py \
    --data_path "/home/msai/adnan002/data/DIHARD3" \
    --type dev \
    --source /home/msai/adnan002/repos/TSVAD_pytorch/ts-vad/models/ecapa-tdnn.model

python /home/msai/adnan002/repos/TSVAD_pytorch/ts-vad/prepare/prepare_dihard.py \
    --data_path "/home/msai/adnan002/data/DIHARD3" \
    --type eval \
    --source /home/msai/adnan002/repos/TSVAD_pytorch/ts-vad/models/ecapa-tdnn.model
