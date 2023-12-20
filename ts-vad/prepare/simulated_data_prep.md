## Simulated Data

The initial structure of the simulated data is as follows:

Use code with caution. Learn more
SIMU3
└── data
    ├── swb_sre_cv_ns1_beta2_200
    ├── ...
    ├── swb_sre_tr_ns6_beta20_1000
└── wav
    ├── swb_sre_cv_ns1_beta2_00
    ├── ...
    └── swb_sre_tr_ns6_beta20_1000

#### To prepare the simulated data, run the following in the below order:

1. Run 1-make_rttm_folders.py
2. Run 2-copy_wav.py
3.1. Run prepare_simulated.sh
3.2. Run 3.2-move_to_all_files.py
4. Run 4-move_jsons.py

#### Your simulated data for TS-VAD should now have the following structure:

all_files
├── rttms
│   ├── data_simu3_wav_swb_sre_cv_ns1_beta2_200_1_mix_0000001.rttm
│   └── < 7200 total files>
├── target_audio
│   ├── data_simu3_wav_swb_sre_cv_ns1_beta2_200_1_mix_0000001
│       ├── 1.wav
│       ├── all.wav
│       └── <eachspeaker.wav + all.wav>
│   └── < 7200 total dirs>
├── target_embedding
│   ├── data_simu3_wav_swb_sre_cv_ns1_beta2_200_1_mix_0000001
│       ├── 1.pt
│       └── <eachspeaker.pt>
│   └── < 7200 total dirs>
└── all_simtrain.json

Once the files are in this structure, this is ready to be used by the TS-VAD module