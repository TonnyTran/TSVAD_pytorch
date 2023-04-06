#!/bin/bash

python main.py \
--train_list /data08/alimeeting/ts_Train.json \
--eval_list /data08/alimeeting/ts_Eval.json \
--train_path /data08/alimeeting/Train_Ali_far \
--eval_path /data08/alimeeting/Eval_Ali_far \
--save_path exps/res1 \
--rs_len 16 \
--test_shift 4 \
--min_silence 0.32 \
--min_speech 0.00 \
--threshold 0.60 \
--n_cpu 12 \
--eval