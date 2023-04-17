# TS-VAD

## Prepare
- Download Alimeeting dataset: Train_Ali_far.tar.gz and Eval_Ali.tar.gz. The dataset looks like:

        # alimeeting (Only far data, the first channel is used)
        # ├── Train_Ali_far 
        # │   ├── audio_dir
        # │   ├── textgrid_dir
        # ├── Eval_Ali_far 
        # │   ├── audio_dir
        # │   ├── textgrid_dir

- [wavlm pretrain model](https://drive.google.com/file/d/1-zlAj2SyVJVsbhifwpTlAfrgc9qu-HDb/view?usp=share_link), put in 'ts-vad/pretrained_models' (build this folder)

### from wespeaaker_alimeeting/run.sh shell
- Use run.sh, step 1 can extract the target audio and embedding, genenerate the json file automaticlly.

### from google drive
- [My json files + extracted speaker embedding + trained ts-vad model + training log](https://drive.google.com/drive/folders/1AFip2h9W7sCFbzzasL_fAkGUNZOzaTGK?usp=share_link), put the ecapa-tdnn.model into 'wespeaker_alimeeting/pretrainde_models', put the ts-vad.model into 'ts-vad/pretrained_models'.


## Dataset Format
My dataset looks like this:

    # alimeeting (Only far data, the first channel is used)
    # ├── Train_Ali_far 
    # │   ├── target_embedding (target speaker embeddings, find in Google Link)
    # │       ├── R0008_M0054_MS002 (video id)
    # │           ├── 1.pt (embeddings for the first speaker)
    # │           ├── 2.pt
    # │           ├── 3.pt
    # │           ├── 4.pt
    # │       ├── ...
    # │   ├── target_speech (speech dataset, obtain by run prepare_data.py)
    # │       ├── R0008_M0054_MS002 (video id)
    # │           ├── 1.wav (clean speech for the first speaker)
    # │           ├── 2.wav
    # │           ├── 3.wav
    # │           ├── 4.wav
    # │           ├── all.wav (the entire speech for the video, only the first channel)
    # │       ├── ...
    # │   ├── audio_dir
    # │   ├── textgrid_dir
    # ├── Eval_Ali_far (Similar)
    # │   ├── ...
    # ├── Pseudo_Ali_far (Generate during clustering based speaker diarization)
    # │   ├── target_embedding (target speaker embeddings)
    # │   ├── target_speech (speech dataset)

## Usage
- [1] bash run.sh in wespeaker_alimeeting, only step 1, to prepare dataset
- [2] bash run_train.sh in ts-vad, obtain the ts-vad model
- TS-VAD Result (Ground Truth): DER / MS / FA / SC = 3.67 / 1.61 / 1.68 / 0.38
- [3] bash run_eval.sh for evaluation (init_model: select the model for evaluation), evalute use ground truth speech
- [4] bash run.sh in wespeaker_alimeeting, step 2 to step 9, can obtain two results:
- Clustering Result: DER / MS / FA / SC = 16.55 / 14.53 / 1.13 / 0.89
- TS-VAD Result (Pseudo Label): DER / MS / FA / SC = 3.64 / 1.56 / 1.69 / 0.38

## Explaination
- Speaker embedding is extracted from ecapa-tdnn model. I train this model on CnCeleb1+2+Alimeeting Training set
- To simply the code, the ground truth number of speakers are used in the wespeaker_alimeeting clustering process