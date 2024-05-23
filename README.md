# TS-VAD on DIHARD3

## Installation
### Clone the repo and create a new virtual environment

Clone the repo:
```
git clone -b adnan https://github.com/TonnyTran/TSVAD_pytorch.git
cd TSVAD_pytorch
```
Using conda to create a fresh virtual environment with the dependencies, and activate it:
```
conda env create --name tsvad --file=tsvad.yaml
conda activate tsvad
```

## Add pre-trained libraries

- Add [WavLM-Base+.pt](https://drive.google.com/file/d/1-zlAj2SyVJVsbhifwpTlAfrgc9qu-HDb/view?usp=share_link) in `ts-vad/pretrained_models` (build this folder)

- Add [ecapa-tdnn.model](https://drive.google.com/file/d/1E-ju12Jy1fID2l4x-nj0zB5XUHSKWsRB/view?usp=drive_link) to `ts-vad/pretrained_models`

- We use musan and rirs for data augmentation and you can pass these parameters in `run_train.sh`. Download [musan](https://www.openslr.org/17/) and [rirs noise](https://www.openslr.org/28/).


## Prepare DIHARD3

- Download DIHARD3 dataset
```
pip install gdown
gdown --folder 1S3RqdbUszN1nRjUAmtm_PvrBDtnXn65Z
```
- The dataset looks like

        # DIHARD3
        # ├── third_dihard_challenge_dev 
        # |   ├──── data
        # │     ├── flac
        # │     ├── rttm
        # |     ├── sad
        # |     ├── uem
        # |     ├── uem_scoring
        # ├── third_dihard_challenge_eval 
        # |   ├──── data
        # │     ├── flac
        # │     ├── rttm
        # |     ├── sad
        # |     ├── uem
        # |     ├── uem_scoring

- Install sox using `sudo apt-get install sox`. Create a folder `wav` in `third_dihard_challenge_dev` and `third_dihard_challenge_eval`. Convert all flac audios to wav and keep it in this folder `wav`.

- For the `wav` directory, you can use this script. This creates the `wav` folder in the same parent directory as `flac` and converts each `.flac` file to `.wav`. Install `sox` using `sudo apt-get install sox` before running the below script:  
```mkdir -p "flac/wav" && find "flac" -type f -name "*.flac" -exec sh -c 'sox "{}" "$(dirname "{}")/wav/$(basename "{}" .flac).wav"' \; && mv "flac/wav" .```.

### Downsampling and Upsampling

For best results, we found that matching the DIHARD3 data with the simulated pretraining data gives better results. Simulated data is upsampled from 8K to 16K.

To mimic 8K to 16K, we first downsample 16K DIHARD3 wav to 8K and then upsample back to 16K. This boosts the TS-VAD results on DIHARD3 significantly.

To make this change simply run below code inside `wav` folders:
```for file in *.wav; do sox -G "$file" -r 8000 temp.wav; sox -G temp.wav -r 16000 "${file%.wav}_converted.wav"; done; rm temp.wav```

DEV: ```rm DH_DEV_*[0-9].wav && rename 's/_converted//' *_converted.wav```  
EVAL: ```rm DH_EVAL_*[0-9].wav && rename 's/_converted//' *_converted.wav```

### Prepare Target Audio and Embeddings

- Execute `ts-vad/prepare/prepare_dihard.sh` to get `target_audio` and `target_embeddings` and `ts_***.json` file

- In `third_dihard_challenge_eval/data/rttm`, run command `cat *.rttm > all.rttm`. This will be used later to get the DER scores. Rest of the rttm files should be replaced with clustering model rttm files.
[NOTE: all.rttm should be from the ground truth (since it is used to calculate DER) and data/rttm should have the clustering model rttm results]

- Following these steps your DIHARD3 directory should have these files and folders to run the TS-VAD model:

        # DIHARD3
        # ├── third_dihard_challenge_dev 
        # |   ├──── data
        # │     ├── wav
        # |         |── DH_DEV_****.wav
        # |         |── ....
        # |         |── DH_DEV_****.wav
        # │     ├── rttm
        # |         |── DH_DEV_****.rttm
        # |         |── ....
        # |         |── DH_DEV_****.rttm
        # |     ├── target_audio
        # │         |── DH_DEV_****
        # │             ├── <eachspeaker.wav> and <all.wav>
        # |         |── ...
        # │         |── DH_DEV_****
        # │             ├── <eachspeaker.wav> and <all.wav>
        # |     ├── target_embeddings
        # │         |── DH_DEV_****
        # │             ├── <eachspeaker.pt>
        # |         |── ...
        # │         |── DH_DEV_****
        # │             ├── <eachspeaker.pt>
        # |     ├── ts_dev.json
        # ├── third_dihard_challenge_eval 
        # |   ├──── data
        # │     ├── wav
        # |         |── DH_EVAL_****.wav
        # |         |── ....
        # |         |── DH_EVAL_****.wav
        # │     ├── rttm
        # |         |── all.rttm
        # |         |── DH_EVAL_****.rttm
        # |         |── ....
        # |         |── DH_EVAL_****.rttm
        # |     ├── target_audio
        # │         |── DH_EVAL_****
        # │             ├── <eachspeaker.wav> and <all.wav>
        # |         |── ...
        # │         |── DH_EVAL_****
        # │             ├── <eachspeaker.wav> and <all.wav>
        # |     ├── target_embeddings
        # │         |── DH_EVAL_****
        # │             ├── <eachspeaker.pt>
        # |         |── ...
        # │         |── DH_EVAL_****
        # │             ├── <eachspeaker.pt>
        # |     ├── ts_eval.json

## Usage
- Edit `run_train.sh` file with correct parameters. You should also pass musan and rirs path here.

- A sample script looks like this:

```
python main.py \
--train_list DIHARD3/third_dihard_challenge_dev/data/ts_dev.json \
--eval_list DIHARD3/third_dihard_challenge_eval/data/ts_eval.json \
--train_path DIHARD3/third_dihard_challenge_dev/data \
--eval_path DIHARD3/third_dihard_challenge_eval/data \
--musan_path data/musan \
--rir_path data/RIRS_NOISES/simulated_rirs \
--save_path exps/res23 \
--warm_up_epoch 10 \
--batch_size 40 \
--rs_len 4 \
--test_shift 4 \
--lr 0.0001 \
--test_step 1 \
--max_epoch 40 \
--train
```

## Simulated Data

### Initial Structure

The initial structure of the simulated data is as follows:

```plaintext
SIMU3
└── data
    ├── swb_sre_cv_ns1_beta2_200
    ├── .....
    └── swb_sre_tr_ns6_beta20_1000 
└── wav 
    ├── swb_sre_cv_ns1_beta2_200
    ├── .....
    └── swb_sre_tr_ns6_beta20_1000 
```

### Preparation Steps

To prepare the simulated data, execute the following scripts in the given order:

1. `1-make_rttm_folders.py`
2. `2-copy_wav.py`
3. `prepare_simulated.sh`
4. `3.2-move_to_all_files.py`
5. `4-move_jsons.py`

### Final Structure

After running the above scripts, your simulated data for TS-VAD should have the following structure:

```plaintext
all_files
├── rttms
│   ├── data_simu3_wav_swb_sre_cv_ns1_beta2_200_1_mix_0000001.rttm
│   └── < 7200 total files>
├── target_audio
│   ├── data_simu3_wav_swb_sre_cv_ns1_beta2_200_1_mix_0000001
│   │   ├── 1.wav
│   │   ├── all.wav
│   │   └── <eachspeaker.wav + all.wav>
│   └── < 7200 total dirs>
├── target_embedding
│   ├── data_simu3_wav_swb_sre_cv_ns1_beta2_200_1_mix_0000001
│   │   ├── 1.pt
│   │   └── <eachspeaker.pt>
│   └── < 7200 total dirs>
└── all_simtrain.json
```

### Usage

Additionally pass `--simtrain True` if using simulated data for training

```
python main.py \
--train_list v2_simulated_data_Switchboard_SRE_small_16k/data/simu3/data/all_files/all_simtrain.json \
--eval_list DIHARD3/third_dihard_challenge_eval/data/ts_eval.json \
--train_path v2_simulated_data_Switchboard_SRE_small_16k/data/simu3/data/all_files \
--eval_path DIHARD3/third_dihard_challenge_eval/data \
--save_path exps/res23 \
--warm_up_epoch 10 \
--batch_size 40 \
--rs_len 4 \
--test_shift 4 \
--lr 0.0001 \
--test_step 1 \
--max_epoch 40 \
--train \
--simtrain True
```

Note: `eval_list` and `eval_path` is not used in `simtrain` mode. If you wish to evaluate as well, you can fix the code around the line `s.eval_network(args)` in `main.py`

### Fine-tuning on DIHARD3

After pre-training on simulated data, pass the trained model on simulated data using `--init_model` parameter. Keep everything for DIHARD3 training the same. This finetunes the trained model on DIHARD3 dev set.

### Results
- Download our best models from [this link](https://entuedu-my.sharepoint.com/:f:/g/personal/adnan002_e_ntu_edu_sg/EnYPxis6jm5Ao8wBRQYDi9sBloQY7T2l52rsxUL-WkF2-g?e=IRcgo7). Password: 1234

- Ground truth DER score
Full Set (c=0.25): `DER 8.97%, MS 5.97%, FA 0.90%, SC 2.10%`
Core Set (c=0.0): `DER 24.27%, MS 12.9%, FA 7.6%, SC 3.8%`

- Clustering DER score
Full Set (c=0.25): `DER 12.45%, MS 4.61%, FA 4.54%, SC 3.30%`
Core Set (c=0.0): `DER 28.74%, MS 15.6%, FA 6.4%, SC 6.7%`

- To replicate the results. Use `--eval` mode and pass `--init_model` from the model in the link above:

```
python main.py \
--train_list data/DIHARD3/third_dihard_challenge_dev/data/ts_dev.json \
--eval_list data/DIHARD3/third_dihard_challenge_eval/data/ts_eval.json \
--train_path data/DIHARD3/third_dihard_challenge_dev/data \
--eval_path data/DIHARD3/third_dihard_challenge_eval/data \
--save_path exps/eval24 \
--rs_len 4 \
--test_shift 0.5 \
--min_silence 0.32 \
--min_speech 0.00 \
--threshold 0.50 \
--n_cpu 12 \
--eval \
--init_model <replace with downloaded model> \
```


## Notice
- Change the data path in ts-vad/run_train.sh and run_eval.sh, include the musan and rir dataset path.
- Speaker embedding is extracted from ecapa-tdnn model. I train this model on CnCeleb1+2+Alimeeting Training set
- You may need to run commands like `chmod +x *.sh` for shell script files on linux.
- For simplicity, you can try with ground truth embeddings. Otherwise, replace `third_dihard_challenge_eval/data/rttm` files with clustering method results or whatever approach you wish to use.
- To use clustering results for TS-VAD, replace rttm folder in `third_dihard_challenge_eval/data/rttm` with your clustering result rttm files.

## Scoring

To check the core results:
- Run `chmod +x <files>`: `parse_options.sh` and `rttm_from_uem.py`
- Specify the DIHARD3 EVAL directory variable inside `./score.sh`
- Copy the `res_rttm` (experiment rttm) and `all_rttm` (correct rttm) into `scoring/` and run `./score.sh`