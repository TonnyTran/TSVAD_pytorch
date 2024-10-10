DATA_PATH="data/DIHARD3vbx8"
OUTPUT_PATH="exps/wavlmbaseplus"

python main.py \
--train_list ${DATA_PATH}/third_dihard_challenge_dev/data/ts_dev.json \
--eval_list ${DATA_PATH}/third_dihard_challenge_eval/data/ts_eval.json \
--train_path ${DATA_PATH}/third_dihard_challenge_dev/data \
--eval_path ${DATA_PATH}/third_dihard_challenge_eval/data \
--musan_path data/musan \
--rir_path data/RIRS_NOISES/simulated_rirs \
--save_path ${OUTPUT_PATH} \
--max_speaker 8 \
--warm_up_epoch 10 \
--batch_size 40 \
--rs_len 4 \
--test_shift 4 \
--lr 0.0001 \
--test_step 1 \
--max_epoch 40 \
--speech_encoder_model microsoft/wavlm-base-plus \
--train