# TS-VAD on DIHARD3

## Prepare
- Download DIHARD3 dataset: The dataset looks like

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

- Create a folder `wav` in `third_dihard_challenge_dev` and `third_dihard_challenge_eval`. Convert all flac audios to wav and keep it in this folder `wav`

- Execute `ts-vad/prepare/prepare_dihard.sh` to get `target_audio` and `target_embeddings` and `ts_***.json` file

- In `third_dihard_challenge_eval/data/rttm`, run command `cat *.rttm > all.rttm`. This will be used later to get the DER scores.

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

## Add pre-trained libraries

- Add [WavLM-Base+.pt](https://drive.google.com/file/d/1-zlAj2SyVJVsbhifwpTlAfrgc9qu-HDb/view?usp=share_link) in `ts-vad/pretrained_models` (build this folder)

- Add `ecapa-tdnn.model` to `ts-vad/pretrained_models`

## Usage
- Edit `run_train.sh` file with correct parameters. You should also pass musan and rirs path here.

- A sample script looks like this:

```
python main.py \
--train_list DIHARD3/third_dihard_challenge_dev/data/ts_dev.json \
--eval_list DIHARD3/third_dihard_challenge_eval/data/ts_eval.json \
--train_path DIHARD3/third_dihard_challenge_dev/data \
--eval_path DIHARD3/third_dihard_challenge_eval/data \
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

## Notice
- Change the data path in wespeaker_alimeeting/run.sh, data_path=/data08/alimeeting
- Change the data path in ts-vad/run_train.sh and run_eval.sh, include the musan and rir dataset path.
- You may need to run commands like `chmod +x *.sh` for shell script files on linux
- Speaker embedding is extracted from ecapa-tdnn model. I train this model on CnCeleb1+2+Alimeeting Training set
- For simplicity, you can try with ground truth embeddings. Otherwise, replace `third_dihard_challenge_eval/data/rttm` files with clustering method results or whatever approach you wish to use.
- You may need to do minor change in `get_ids()` in dataLoader.py