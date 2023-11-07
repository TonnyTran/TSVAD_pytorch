#!/bin/bash
#SBATCH --partition=SCSEGPU_M1 
#SBATCH --qos=q_amsai 
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1 
#SBATCH --mem=1G 
#SBATCH --job-name=eval-dihard-core-6s
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

python main.py \
--train_list /home/msai/adnan002/data/DIHARD3/third_dihard_challenge_dev/data/ts_dev.json \
--eval_list /home/msai/adnan002/data/DIHARD3/third_dihard_challenge_eval/data/ts_eval.json \
--train_path /home/msai/adnan002/data/DIHARD3/third_dihard_challenge_dev/data \
--eval_path /home/msai/adnan002/data/DIHARD3/third_dihard_challenge_eval/data \
--save_path exps/eval24 \
--rs_len 4 \
--test_shift 0.5 \
--min_silence 0.32 \
--min_speech 0.00 \
--threshold 0.50 \
--n_cpu 12 \
--eval \
--init_model /home/msai/adnan002/repos/TSVAD_pytorch/ts-vad/exps/res23/model/model_0024.model \