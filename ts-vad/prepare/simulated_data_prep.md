# Simulated Data

## Initial Structure

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

## Preparation Steps

To prepare the simulated data, execute the following scripts in the given order:

1. `1-make_rttm_folders.py`
2. `2-copy_wav.py`
3. `prepare_simulated.sh`
4. `3.2-move_to_all_files.py`
5. `4-move_jsons.py`

## Final Structure

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

## Usage

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