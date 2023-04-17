#!/bin/bash

python main.py \
--train_list /data08/alimeeting/Train_Ali_far/ts_Train.json \
--eval_list /data08/alimeeting/Eval_Ali_far/ts_Eval.json \
--train_path /data08/alimeeting/Train_Ali_far \
--eval_path /data08/alimeeting/Pseudo_Ali_far \
--save_path exps/debug1 \
--rs_len 4 \
--test_shift 1 \
--min_silence 0.32 \
--min_speech 0.00 \
--threshold 0.70 \
--n_cpu 12 \
--eval \
--init_model pretrained_models/ts-vad.model \