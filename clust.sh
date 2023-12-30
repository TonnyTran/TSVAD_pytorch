stage=1
stop_stage=6

data_path="/home/users/ntu/tlkushag/scratch/data08/alimeeting"   # The path of the dataset

eval_path=${data_path}/Eval_Ali_far
pseudo_path=${data_path}/Pseudo_Ali_far
audio_dir=${eval_path}/audio_dir
textgrid_dir=${eval_path}/textgrid_dir
eval_json=${eval_path}/ts_Eval.json
target_audio_path=${pseudo_path}/target_audio
target_embedding_path=${pseudo_path}/target_embedding

current_path=`pwd`
clustering_path=${current_path}/clustering
res_rttm_path=./exp/predict/res_rttm

stage1() {
    echo "[1] Process dataset: Train/Eval dataset, get target speech and emebdding, get json files"
    mkdir -p ./exp/predict
    ls ${audio_dir}/*.wav | awk -F/ '{print substr($NF, 1, length($NF)-4), $0}' > ./exp/predict/wav.scp
    python ./clustering/modules/prepare_data.py \
        --data_path ${data_path} \
        --type Eval \
        --source ./pretrained_models/ecapa-tdnn.model
    python ./clustering/modules/prepare_data.py \
        --data_path ${data_path} \
        --type Train \
        --source ./pretrained_models/ecapa-tdnn.model
}

stage2() {
    min_duration=0.255
    echo "[2] VAD"
    python ./clustering/modules/vad.py \
            --repo-path ./clustering/external_tools/silero-vad-3.1 \
            --scp ./exp/predict/wav.scp \
            --threshold 0.2 \
            --min-duration $min_duration > ./exp/predict/vad 
        }

stage3() {
    echo "[3] Extract and cluster"
    python ./clustering/modules/cluster.py \
            --scp ./exp/predict/wav.scp \
            --segments ./exp/predict/vad \
            --source ./pretrained_models/ecapa-tdnn.model \
            --output ./exp/predict/vad_labels
            
}

stage4() {
    echo "[4] Get RTTM"
    python ./clustering/modules/make_rttm.py \
            --labels ./exp/predict/vad_labels \
            --channel 1 > ${res_rttm_path}            
}

stage5() {
    echo "[5] Get labels"
    mkdir -p ./exp/tmp
    mkdir -p ./exp/label
    find ${audio_dir} -name "*\.wav" > ./exp/tmp/tmp
    sort  ./exp/tmp/tmp > ./exp/tmp/wavlist
    awk -F '/' '{print $NF}'  ./exp/tmp/wavlist | awk -F '.' '{print $1}' >  ./exp/tmp/uttid
    find -L $textgrid_dir -iname "*.TextGrid" >  ./exp/tmp/tmp
    sort  ./exp/tmp/tmp  > ./exp/tmp/textgrid.flist
    paste ./exp/tmp/uttid ./exp/tmp/textgrid.flist > ./exp/tmp/uttid_textgrid.flist
    while read text_file
    do
        text_grid=`echo $text_file | awk '{print $1}'`
        text_grid_path=`echo $text_file | awk '{print $2}'`
        python ./clustering/external_tools/make_textgrid_rttm.py --input_textgrid_file $text_grid_path \
                                        --uttid $text_grid \
                                        --output_rttm_file ./exp/label/${text_grid}.rttm
    done < ./exp/tmp/uttid_textgrid.flist
    rm -r ./exp/tmp
    cat ./exp/label/*.rttm > ./exp/label/all.rttm    
}

stage6() {
    echo "[6] Get DER result"
    perl ./clustering/external_tools/SCTK-2.4.12/src/md-eval/md-eval.pl -c 0.25 -r ./exp/label/all.rttm -s ${res_rttm_path} 
}

# Run the stages in a loop
for ((i = $stage; i <= $stop_stage; i++)); do
    stage${i}  # Call the corresponding stage function
done
