# python prepare_dihard.py \
#     --data_path "/home/users/ntu/adnan002/scratch/data/DIHARD3" \
#     --type dev \
#     --max_speaker 8 \
#     --source ../pretrained_models/ecapa-tdnn.model

python prepare_dihard.py \
    --data_path "/home/users/ntu/adnan002/scratch/data/DIHARD3" \
    --type eval \
    --max_speaker 8 \
    --source ../pretrained_models/ecapa-tdnn.model
