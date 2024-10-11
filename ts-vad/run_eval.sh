DATA_PATH="/workspace/TSVAD_pytorch/ts-vad/data/alimeeting"
OUTPUT_PATH="exps/eval24_31"
MODEL_PATH="/workspace/TSVAD_pytorch/ts-vad/exps/res24/model/model_0031.model"

python main.py \
--train_list ${DATA_PATH}/Train_Ali_far/ts_Train.json \
--eval_list ${DATA_PATH}/Eval_Ali_far/ts_Eval.json \
--train_path ${DATA_PATH}/Train_Ali_far \
--eval_path ${DATA_PATH}/Eval_Ali_far \
--save_path ${OUTPUT_PATH} \
--rs_len 4 \
--test_shift 0.5 \
--min_silence 0.32 \
--min_speech 0.00 \
--threshold 0.50 \
--n_cpu 12 \
--eval \
--init_model ${MODEL_PATH} \
--max_speaker 4