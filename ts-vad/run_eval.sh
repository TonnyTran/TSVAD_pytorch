#!/bin/bash
dihard_path=/home/users/ntu/tlkushag/scratch/data08/dihard
curr_path=/home/users/ntu/tlkushag/scratch/TSVAD_pytorch/ts-vad

python ${curr_path}/main.py \
--train_list ${dihard_path}/third_dihard_challenge_dev/data/ts_dev.json \
--eval_list ${dihard_path}/third_dihard_challenge_eval/data/ts_eval.json \
--train_path ${dihard_path}/third_dihard_challenge_dev/data \
--eval_path ${dihard_path}/third_dihard_challenge_eval/data \
--save_path ${curr_path}/exps/res23 \
--rs_len 4 \
--test_shift 0.5 \
--min_silence 0.32 \
--min_speech 0.00 \
--threshold 0.50 \
--n_cpu 4 \
--eval \
--init_model /home/users/ntu/tlkushag/scratch/TSVAD_pytorch/ts-vad/exps/res23/model/clust/model_0024.model \