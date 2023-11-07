#!/bin/bash
#SBATCH --partition=SCSEGPU_M1 
#SBATCH --qos=q_amsai 
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1 
#SBATCH --mem=1G 
#SBATCH --job-name=cluster-dihard-3-6
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

stage=5
stop_stage=6
data_path=/home/msai/adnan002/data/DIHARD3
eval_path=${data_path}/third_dihard_challenge_eval/data
# pseudo_path=${data_path}/Pseudo_Ali_far
audio_dir=${eval_path}/wav
rttm_dir=${eval_path}/rttm
eval_json=${eval_path}/ts_Eval.json
# target_audio_path=${pseudo_path}/target_audio
# target_embedding_path=${pseudo_path}/target_embedding

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "[1] Process dataset: Train/Eval dataset, get target speech and emebdding, get json files"
    mkdir -p exp/predict
    ls ${audio_dir}/*.wav | awk -F/ '{print substr($NF, 1, length($NF)-4), $0}' > exp/predict/wav.scp
    python modules/prepare_data.py \
        --data_path ${data_path} \
        --type Eval \
        --source pretrained_models/ecapa-tdnn.model
    python modules/prepare_data.py \
        --data_path ${data_path} \
        --type Train \
        --source pretrained_models/ecapa-tdnn.model
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    min_duration=0.255
    echo "[2] VAD"
    python modules/vad.py \
            --repo-path external_tools/silero-vad-3.1 \
            --scp exp/predict/wav.scp \
            --threshold 0.2 \
            --min-duration $min_duration > exp/predict/vad
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "[3] Extract and cluster"
    python modules/cluster.py \
            --scp exp/predict/wav.scp \
            --segments exp/predict/vad \
            --source /home/msai/adnan002/repos/TSVAD_pytorch/ts-vad/models/ecapa-tdnn.model \
            --output exp/predict/vad_labels
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "[4] Get RTTM"
    python modules/make_rttm.py \
            --labels exp/predict/vad_labels \
            --channel 1 > exp/predict/res_rttm
fi

if [ $stage -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "[5] Get labels"
    mkdir -p exp/tmp
    mkdir -p exp/label
    find ${audio_dir} -name "*\.wav" > exp/tmp/tmp
    sort  exp/tmp/tmp > exp/tmp/wavlist
    awk -F '/' '{print $NF}'  exp/tmp/wavlist | awk -F '.' '{print $1}' >  exp/tmp/uttid
    find -L $textgrid_dir -iname "*.rttm" >  exp/tmp/tmp
    sort  exp/tmp/tmp  > exp/tmp/textgrid.flist
    paste exp/tmp/uttid exp/tmp/textgrid.flist > exp/tmp/uttid_textgrid.flist
    while read text_file
    do
        text_grid=`echo $text_file | awk '{print $1}'`
        text_grid_path=`echo $text_file | awk '{print $2}'`
        # python external_tools/make_textgrid_rttm.py --input_textgrid_file $text_grid_path \
        #                                --uttid $text_grid \
        #                                --output_rttm_file exp/label/${text_grid}.rttm
    done < exp/tmp/uttid_textgrid.flist
    rm -r exp/tmp
    cat exp/label/*.rttm > exp/label/all.rttm
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "[6.1] Combine rttm files"
    cat /home/msai/adnan002/data/DIHARD3/third_dihard_challenge_eval/data/rttm/*.rttm > exp/label/all.rttm
    echo "[6.2] Get DER result"
    perl external_tools/SCTK-2.4.12/src/md-eval/md-eval.pl -c 0.25 -r exp/label/all.rttm -s exp/predict/res_rttm 
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "[7] Get target speech"
    if [ -d "${target_audio_path}" ]; then
        rm -r ${target_audio_path}
    fi
    python modules/extract_target_speech.py \
        --rttm_path exp/predict/res_rttm \
        --orig_audio_path ${audio_dir} \
        --target_audio_path ${target_audio_path}
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "[8] Get target embeddings"
    if [ -d "${target_embedding_path}" ]; then
        rm -r ${target_embedding_path}
    fi
    python modules/extract_target_embedding.py \
        --target_audio_path ${target_audio_path} \
        --target_embedding_path ${target_embedding_path} \
        --source pretrained_models/ecapa-tdnn.model
fi

# if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
#     echo "[9] Do TS-VAD with pseudo speech"
#     cd ../ts-vad
#     python main.py \
#     --eval_list ${eval_json} \
#     --eval_path ${pseudo_path} \
#     --save_path exps/debug \
#     --rs_len 4 \
#     --test_shift 1 \
#     --threshold 0.50 \
#     --n_cpu 12 \
#     --eval \
#     --init_model ../ts-vad/exps/res23/model/model_0036.model
#     cd -
# fi

# if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
#     echo "Round2"
#     echo "Get target speech again"
#     if [ -d "${target_audio_path}" ]; then
#         rm -r ${target_audio_path}
#     fi
#     python modules/extract_target_speech.py \
#         --rttm_path ../ts-vad/exps/debug/res_rttm \
#         --orig_audio_path ${audio_dir} \
#         --target_audio_path ${target_audio_path}
# fi

# if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
#     echo "Get target embeddings"
#     if [ -d "${target_embedding_path}" ]; then
#         rm -r ${target_embedding_path}
#     fi
#     python modules/extract_target_embedding.py \
#         --target_audio_path ${target_audio_path} \
#         --target_embedding_path ${target_embedding_path} \
#         --source pretrained_models/ecapa-tdnn.model \
#         --length_embedding 6 \
#         --step_embedding 1 \
#         --batch_size 96
# fi

# if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
#     echo "Do TS-VAD with pseudo speech"
#     cd ../ts-vad
#     python main.py \
#     --eval_list ${eval_json} \
#     --eval_path ${pseudo_path} \
#     --save_path exps/debug \
#     --rs_len 4 \
#     --test_shift 4 \
#     --threshold 0.60 \
#     --n_cpu 12 \
#     --eval \
#     --init_model pretrain/ts-vad.model
#     cd -
# fi