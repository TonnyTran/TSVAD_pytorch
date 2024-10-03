DATA_PATH="data/DIHARD3vbx8"
OUTPUT_PATH="exps/eval24"
MODEL_PATH="pretrained_models/newlongsimdata_pretrain_sep_24/p8new_15_19eval/model/model_0019_finetuned.model"

python main.py \
--train_list ${DATA_PATH}/third_dihard_challenge_dev/data/ts_dev.json \
--eval_list ${DATA_PATH}/third_dihard_challenge_eval/data/ts_eval.json \
--train_path ${DATA_PATH}/third_dihard_challenge_dev/data \
--eval_path ${DATA_PATH}/third_dihard_challenge_eval/data \
--save_path ${OUTPUT_PATH} \
--rs_len 4 \
--test_shift 0.5 \
--min_silence 0.32 \
--min_speech 0.00 \
--threshold 0.50 \
--n_cpu 12 \
--eval \
--init_model ${MODEL_PATH} \
--max_speaker 8