# TS-VAD on AliMeeting

This guide will walk you through setting up the TSVAD_pytorch environment and downloading necessary data.

## 1. Conda Environment Setup

```bash
cd /mnt
rm -rf miniconda3/

INSTALLER="./Miniconda3-latest-Linux-x86_64.sh"

if [ ! -f "$INSTALLER" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
else
    echo "Installer $INSTALLER already exists."
fi

INSTALL_DIR="/mnt/miniconda3"

bash "$INSTALLER" -b -p "$INSTALL_DIR"
export PATH="/mnt/miniconda3/bin:$PATH"
conda env create --name wespeak2 --file=/mnt/wespeak2.yml
source activate wespeak2
```

## 2. Pull from GitHub

```bash
git clone https://github.com/adnan-azmat/TSVAD_pytorch.git
cd TSVAD_pytorch/ts-vad
git checkout u/adnan/alimeeting
mkdir pretrained_models
cd pretrained_models

gdown 1-zlAj2SyVJVsbhifwpTlAfrgc9qu-HDb # [WavLM-Base+.pt](https://drive.google.com/file/d/1-zlAj2SyVJVsbhifwpTlAfrgc9qu-HDb/view?usp=share_link)
gdown 1E-ju12Jy1fID2l4x-nj0zB5XUHSKWsRB # [ecapa-tdnn.model](https://drive.google.com/file/d/1E-ju12Jy1fID2l4x-nj0zB5XUHSKWsRB/view?usp=drive_link)
```

## 3. Prepare Datasets

```bash
cd ..
mkdir data
cd data

# MUSAN and RIRS_NOISES (https://www.openslr.org/17/) and (https://www.openslr.org/28/)
wget https://us.openslr.org/resources/17/musan.tar.gz
wget https://us.openslr.org/resources/28/rirs_noises.zip

tar xf musan.tar.gz
sudo apt-get install unzip
unzip rirs_noises.zip

rm musan.tar.gz
rm rirs_noises.zip

# AliMeeting Dataset (https://www.openslr.org/119/)

mkdir alimeeting
cd alimeeting

wget https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Train_Ali_far.tar.gz
tar xf Train_Ali_far.tar.gz
rm Train_Ali_far.tar.gz

wget https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Eval_Ali.tar.gz
tar xf Eval_Ali.tar.gz
rm Eval_Ali.tar.gz
rm -rf Eval_Ali/Eval_Ali_near
mv Eval_Ali/Eval_Ali_far/ .
rm -rf Eval_Ali
```

## 4. Prepare Target Audio and Embeddings

a. Use `ts-vad/prepare/alimeeting/make_textgrid_rttm.py` to create rttm files for AliMeeting

b. Prepare target audio and target embeddings using `ts-vad/prepare/alimeeting/prepare_alimeeting.sh`

Alternatively you can use below script to download pre-computed target embeddings

```bash
gdown 1qM5bGnkYAQMQGhAVIn2R1IbrD8gtKlrH
tar xf alimeeting_embedding.tar
mv embedding/train_target_embedding Train_Ali_far/target_embedding
mv embedding/ts_Train.json Train_Ali_far/ts_train.json
mv embedding/eval_target_embedding Eval_Ali_far/target_embedding
mv embedding/ts_Eval.json Eval_Ali_far/ts_eval.json
rm alimeeting_embedding.tar
rm -rf embedding
```

Your data directory should look like this:
```
ts-vad
└── data
    └── alimeeting
        ├── Eval_Ali_far
        │   ├── audio_dir
        │   ├── rttm_dir
        │   ├── target_audio
        │   ├── target_embedding
        │   ├── textgrid_dir
        │   ├── all.rttm
        │   └── ts_Eval.json
        └── Train_Ali_far
            ├── audio_dir
            │   ├── R0003_M0046_MS002.wav
            │   └── R0003_M0047_MS006.wav
            ├── rttm_dir
            │   ├── R0003_M0046.rttm
            │   └── R0003_M0047.rttm
            ├── target_audio
            │   ├── R0003_M0046_MS002
            │   └── R0003_M0047_MS006
            ├── target_embedding
            │   ├── R0003_M0046_MS002
            │   └── R0003_M0047_MS006
            ├── textgrid_dir
            │   ├── R0003_M0046.TextGrid
            │   └── R0003_M0047.TextGrid
            └── ts_Train.json
```
## 5. Run TS-VAD

Use the `run_train.sh` script in `ts-vad` to train TS-VAD

Use the `run_eval.sh` script in `ts-vad` to evaluate using an existing model
