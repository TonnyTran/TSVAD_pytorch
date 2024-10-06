DATA_PATH="data/alimeeting"
OUTPUT_PATH="exps/eval24"

# # # OPTIONAL: PRE-TRAINED 8 speaker 16 kHz v1 and DIHARD3 FINETUNED MODEL
# # gdown 1h8X9GNkbW_eJJmttMeslz_3mkbLuM4v4
# # sudo apt-get install unzip
# # unzip newlongsimdata_pretrain_sep_24.zip
# # rm newlongsimdata_pretrain_sep_24.zip

MODEL_PATH="pretrained_models/newlongsimdata_pretrain_sep_24/p8new_15_19eval/model/model_0019_finetuned.model"

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