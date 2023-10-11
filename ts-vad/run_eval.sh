#!/bin/bash

python main.py \
--train_list /home/users/ntu/adnan002/scratch/alimeeting/Train_Ali_far/ts_Train.json \
--eval_list /home/users/ntu/adnan002/scratch/alimeeting/Eval_Ali_far/ts_Eval.json \
--train_path /home/users/ntu/adnan002/scratch/alimeeting/Train_Ali_far \
--eval_path /home/users/ntu/adnan002/scratch/alimeeting/Eval_Ali_far \
--save_path exps/res23 \
--rs_len 4 \
--test_shift 0.5 \
--min_silence 0.32 \
--min_speech 0.00 \
--threshold 0.50 \
--n_cpu 12 \
--eval \
--init_model /home/users/ntu/adnan002/scratch/TSVAD_pytorch/ts-vad/pretrained_models/ts-vad.model \