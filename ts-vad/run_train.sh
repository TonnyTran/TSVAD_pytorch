#!/bin/bash
#SBATCH --partition=SCSEGPU_M1 
#SBATCH --qos=q_amsai 
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 
#SBATCH --mem=10G 
#SBATCH --job-name=simdata_train
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err
python /home/msai/adnan002/repos/TSVAD_pytorch/ts-vad/main.py \
--train_list /home/msai/adnan002/data/simulated_data_SD/data/all_files/all_simtrain.json \
--eval_list /home/msai/adnan002/data/DIHARD3/third_dihard_challenge_eval/data/ts_eval.json \
--train_path /home/msai/adnan002/data/simulated_data_SD/data/all_files \
--eval_path /home/msai/adnan002/data/DIHARD3/third_dihard_challenge_eval/data \
--save_path exps/res23 \
--warm_up_epoch 10 \
--batch_size 40 \
--rs_len 4 \
--test_shift 4 \
--lr 0.0001 \
--test_step 5 \
--max_epoch 50 \
--train