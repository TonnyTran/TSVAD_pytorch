python prepare_dihard.py \
    --data_path "data/DIHARD3" \
    --type dev \
    --source ts-vad/pretrained_models/ecapa-tdnn.model

python prepare_dihard.py \
    --data_path "data/DIHARD3" \
    --type eval \
    --source ts-vad/pretrained_models/ecapa-tdnn.model
