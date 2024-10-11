DATA_PATH="/workspace/TSVAD_pytorch/ts-vad/data/alimeeting"
OUTPUT_PATH="exps/res24"
MUSAN_PATH="/workspace/TSVAD_pytorch/ts-vad/data/musan"
RIRS_PATH="/workspace/TSVAD_pytorch/ts-vad/data/RIRS_NOISES/simulated_rirs"

python main.py \
--train_list ${DATA_PATH}/Train_Ali_far/ts_Train.json \
--eval_list ${DATA_PATH}/Eval_Ali_far/ts_Eval.json \
--train_path ${DATA_PATH}/Train_Ali_far \
--eval_path ${DATA_PATH}/Eval_Ali_far \
--musan_path ${MUSAN_PATH} \
--rir_path ${RIRS_PATH} \
--save_path ${OUTPUT_PATH} \
--max_speaker 4 \
--warm_up_epoch 10 \
--batch_size 40 \
--rs_len 4 \
--test_shift 4 \
--lr 0.0001 \
--test_step 1 \
--max_epoch 40 \
--train