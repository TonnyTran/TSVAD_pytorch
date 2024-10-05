python prepare_alimeeting.py \
    --data_path /mnt/TSVAD_pytorch/ts-vad/data/alimeeting/ \
    --type Train \
    --source /mnt/TSVAD_pytorch/ts-vad/pretrained_models/ecapa-tdnn.model

python prepare_alimeeting.py \
    --data_path /mnt/TSVAD_pytorch/ts-vad/data/alimeeting/ \
    --type Eval \
    --source /mnt/TSVAD_pytorch/ts-vad/pretrained_models/ecapa-tdnn.model