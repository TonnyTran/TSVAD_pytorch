#!/bin/bash

python main.py \
--train_list /data08/alimeeting/ts_Train.json \
--eval_list /data08/alimeeting/ts_Eval.json \
--train_path /data08/alimeeting/Train_Ali_far \
--eval_path /data08/alimeeting/Eval_Ali_far \
--save_path exps/res1 \
--warm_up_epoch 10 \
--lr 0.0001 \
--train
