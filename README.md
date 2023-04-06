# TS-VAD

## Prepare
- [My json files + extracted speaker embedding + trained ts-vad model + training log](https://drive.google.com/drive/folders/1AFip2h9W7sCFbzzasL_fAkGUNZOzaTGK?usp=share_link)
- [wavlm pretrain model](https://drive.google.com/file/d/1-zlAj2SyVJVsbhifwpTlAfrgc9qu-HDb/view?usp=share_link)

## Dataset Format
My dataset looks like this:

    # alimeeting (Only far data, the first channel is used)
    # ├── Train_Ali_far 
    # │   ├── ecapa_feature_dir (target speaker embeddings, find in Google Link)
    # │       ├── R0008_M0054_MS002 (video id)
    # │           ├── 1.pt (embeddings for the first speaker)
    # │           ├── 2.pt
    # │           ├── 3.pt
    # │           ├── 4.pt
    # │       ├── ...
    # │   ├── target_audio_dir (speech dataset, obtain by run prepare_data.py)
    # │       ├── R0008_M0054_MS002 (video id)
    # │           ├── 1.wav (clean speech for the first speaker)
    # │           ├── 2.wav
    # │           ├── 3.wav
    # │           ├── 4.wav
    # │           ├── all.wav (the entire speech for the video, only the first channel)
    # │       ├── ...
    # │   ├── audio_dir (not used)
    # │   ├── textgrid_dir (not used)
    # │   ├── resnet_feature_dir (not used)
    # ├── Eval_Ali_far (Similar)
    # │   ├── ...

## Usage
- run prepare_data.py to obatin the all.wav files
- bash run_train.sh for training
- bash run_eval.sh for evaluation (init_model: select the model for evaluation)

- Best Result: DER / MS / FA / SC= 7.34 / 3.21 / 2.19 / 1.94
- You can use bash run_eval.sh to check this result, set test_shift=4

## Explaination
- Speaker embedding is extracted from ecapa-tdnn model. I train this model on CnCeleb1+2+Alimeeting Training set
- Current system uses ground-truth speaker embedding, silence part is not predicted or trained by TS-VAD

## Difference with last version
- Fix the bug for position embedding (Very faster than before, get some improvements)
- Retrain the speaker embedding on CnCeleb1+2+Alimeeting (Similar)
- Offline speaker embedding (1 point improvement)
- Random the start frame for training
- Remove the augmentation
- Remove the ResNet model
- Add some arguments for debugging
- Longer input (16s instead of 6s), Shorter Warm up epoch (10 is enough)
- ... Others, I forget