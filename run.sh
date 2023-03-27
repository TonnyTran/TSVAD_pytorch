#!/bin/bash

python main.py \
--train_list /data08/alimeeting/ts_Train.json \
--eval_list /data08/alimeeting/ts_Eval.json \
--train_path /data08/alimeeting/Train_Ali_far/target_audio_dir \
--eval_path /data08/alimeeting/Eval_Ali_far/target_audio_dir \
--save_path exps/exp6 \
--train
