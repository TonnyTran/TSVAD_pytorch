## Simulated Data

The initial structure of the simulated data is as follows:

Use code with caution. Learn more
SIMU3\
└── data\
&emsp;└── swb_sre_cv_ns1_beta2_200\
&emsp;└── ...\
&emsp;└── swb_sre_tr_ns6_beta20_1000\
└── wav\
&emsp;└── swb_sre_cv_ns1_beta2_00\
&emsp;└── ...\
&emsp;└── swb_sre_tr_ns6_beta20_1000

#### To prepare the simulated data, run the following in the below order:

1. Run 1-make_rttm_folders.py
2. Run 2-copy_wav.py
3.1. Run prepare_simulated.sh
3.2. Run 3.2-move_to_all_files.py
4. Run 4-move_jsons.py

#### Your simulated data for TS-VAD should now have the following structure:

all_files\
└── rttms\
&emsp;└── data_simu3_wav_swb_sre_cv_ns1_beta2_200_1_mix_0000001.rttm\
&emsp;└── < 7200 total files>\
└── target_audio\
&emsp;└── data_simu3_wav_swb_sre_cv_ns1_beta2_200_1_mix_0000001\
&emsp;&emsp;└── 1.wav\
&emsp;&emsp;└── all.wav\
&emsp;&emsp;└── <eachspeaker.wav + all.wav>\
&emsp;└── < 7200 total dirs>\
└── target_embedding\
&emsp;└── data_simu3_wav_swb_sre_cv_ns1_beta2_200_1_mix_0000001\
&emsp;&emsp;└── 1.pt\
&emsp;&emsp;└── <eachspeaker.pt>\
&emsp;└── < 7200 total dirs>\
└── all_simtrain.json

Once the files are in this structure, this is ready to be used by the TS-VAD module
