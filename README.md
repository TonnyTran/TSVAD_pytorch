# TS-VAD on AliMeeting

This guide will walk you through setting up the TSVAD_pytorch environment and downloading necessary data.

## 1. Conda Environment Setup

```bash
cd /workspace
rm -rf miniconda3/

INSTALLER="./Miniconda3-latest-Linux-x86_64.sh"

if [ ! -f "$INSTALLER" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
else
    echo "Installer $INSTALLER already exists."
fi

INSTALL_DIR="/workspace/miniconda3"

bash "$INSTALLER" -b -p "$INSTALL_DIR"
export PATH="/workspace/miniconda3/bin:$PATH"
conda env create --name wespeak2 --file=/workspace/wespeak2.yml
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

a. To get target audio and embeddings for Train_Ali_far you can use the below script:

```bash
# Set correct paths
data_path=/workspace/TSVAD_pytorch/ts-vad/data/alimeeting
ecapa_path=/workspace/TSVAD_pytorch/ts-vad/pretrained_models/ecapa-tdnn.model

cd wespeaker_alimeeting

python modules/prepare_data.py \
    --data_path ${data_path} \
    --type Train \
    --source ${ecapa_path}
fi
```

b. For Eval_Ali_far we first need to get the clustering results and then generate target audio and embeddings.

This can be done by running the script file here: `wespeaker_alimeeting/run.sh` . Note: Set the `data_path` and `ecapa_path` before running the script

The score of the wespeaker clustering should be: `DER / MS / FA / SC = 16.54 / 14.53 / 1.13 / 0.88` (Collor Size: 0.25)

Your data directory should look like this:
```
ts-vad
└── data
    └── alimeeting
        ├── Eval_Ali_far
        │   ├── audio_dir
        │   │   ├── ...
        │   │   └── (many .wav files)
        │   ├── rttm_dir
        │   │   ├── ...
        │   │   └── (many .rttm files)
        │   ├── target_audio
        │   │   ├── ...
        │   │   └── (many directories)
        │   ├── target_embedding
        │   │   ├── ...
        │   │   └── (many directories)
        │   ├── textgrid_dir
        │   │   ├── ...
        │   │   └── (many .TextGrid files)
        │   ├── all.rttm
        │   └── ts_Eval.json
        └── Train_Ali_far
            ├── audio_dir
            │   ├── R0003_M0046_MS002.wav
            │   ├── R0003_M0047_MS006.wav
            │   ├── ...
            │   └── (many more .wav files)
            ├── rttm_dir
            │   ├── R0003_M0046.rttm
            │   ├── R0003_M0047.rttm
            │   ├── ...
            │   └── (many more .rttm files)
            ├── target_audio
            │   ├── R0003_M0046_MS002
            │   │   │   ├── 1.wav
            │   │   │   ├── 2.wav
            │   │   │   ├── 3.wav
            │   │   │   └── 4.wav
            │   │   │   └── all.wav
            │   ├── R0003_M0047_MS006
            │   │   │   ├── 1.wav
            │   │   │   ├── 2.wav
            │   │   │   ├── 3.wav
            │   │   │   └── 4.wav
            │   │   │   └── all.wav
            │   ├── ...
            │   └── (many more directories)
            ├── target_embedding
            │   ├── R0003_M0046_MS002
            │   │   │   ├── 1.pt
            │   │   │   ├── 2.pt
            │   │   │   ├── 3.pt
            │   │   │   └── 4.pt
            │   ├── R0003_M0047_MS006
            │   │   │   ├── 1.pt
            │   │   │   ├── 2.pt
            │   │   │   ├── 3.pt
            │   │   │   └── 4.pt
            │   ├── ...
            │   └── (many more directories)
            ├── textgrid_dir
            │   ├── R0003_M0046.TextGrid
            │   ├── R0003_M0047.TextGrid
            │   ├── ...
            │   └── (many more .TextGrid files)
            └── ts_Train.json
```
## 5. Run TS-VAD

To run TS-VAD model on alimeeting dataset

Use the `run_train.sh` script in `ts-vad` to train TS-VAD
The expected result is: `DER 4.58%, MS 2.94%, FA 1.13%, SC 0.51%`


Use the `run_eval.sh` script in `ts-vad` to evaluate using an existing trained model
