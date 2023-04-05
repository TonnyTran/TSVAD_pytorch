# TS-VAD

## Prepare
- [My json files + extracted speaker embedding + trained ts-vad model + training log](https://drive.google.com/drive/folders/1AFip2h9W7sCFbzzasL_fAkGUNZOzaTGK?usp=share_link)
- [wavlm pretrain model](https://drive.google.com/file/d/1-zlAj2SyVJVsbhifwpTlAfrgc9qu-HDb/view?usp=share_link)

## Usage
- run prepare_data.py to obatin the all.wav files
- bash run_train.sh for training
- bash run_eval.sh for evaluation

- best Result: DER / MS / FA / SC= 8.58 / 3.18 / 3.82 / 1.58
- You can use bash run_eval.sh to check this result, set test_shift=2 

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