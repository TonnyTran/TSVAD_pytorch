stage=2
stop_stage=8

export PATH="/workspace/miniconda3/bin:$PATH"
source activate wespeak2

data_path=/workspace/TSVAD_pytorch/ts-vad/data/alimeeting
ecapa_path=/workspace/TSVAD_pytorch/ts-vad/pretrained_models/ecapa-tdnn.model

eval_path=${data_path}/Eval_Ali_far
audio_dir=${eval_path}/audio_dir
textgrid_dir=${eval_path}/textgrid_dir
eval_json=${eval_path}/ts_Eval.json
target_audio_path=${eval_path}/target_audio
target_embedding_path=${eval_path}/target_embedding

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    mkdir -p exp/predict
    ls ${audio_dir}/*.wav | awk -F/ '{print substr($NF, 1, length($NF)-4), $0}' > exp/predict/wav.scp
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
            --source ${ecapa_path} \
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
    find -L $textgrid_dir -iname "*.TextGrid" >  exp/tmp/tmp
    sort  exp/tmp/tmp  > exp/tmp/textgrid.flist
    paste exp/tmp/uttid exp/tmp/textgrid.flist > exp/tmp/uttid_textgrid.flist
    while read text_file
    do
        text_grid=`echo $text_file | awk '{print $1}'`
        text_grid_path=`echo $text_file | awk '{print $2}'`
        echo "${text_grid}"
        echo "${text_grid_path}"
        python external_tools/make_textgrid_rttm.py --input_textgrid_file $text_grid_path \
                                        --uttid $text_grid \
                                        --output_rttm_file exp/label/${text_grid}.rttm
    done < exp/tmp/uttid_textgrid.flist
    rm -r exp/tmp
    cat exp/label/*.rttm > exp/label/all.rttm

    cp exp/label/all.rttm ${eval_path}
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then    
    echo "[6] Get DER result"
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
        --target_audio_path ${target_audio_path} \
        --output_json ${eval_path}/ts_Eval.json
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "[8] Get target embeddings"
    if [ -d "${target_embedding_path}" ]; then
        rm -r ${target_embedding_path}
    fi
    python modules/extract_target_embedding.py \
        --target_audio_path ${target_audio_path} \
        --target_embedding_path ${target_embedding_path} \
        --source ${ecapa_path}
fi