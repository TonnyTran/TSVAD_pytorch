#!/bin/bash

input_dir="/mnt/TSVAD_pytorch/ts-vad/data/alimeeting/Eval_Ali_far/textgrid_dir"
output_dir="/mnt/TSVAD_pytorch/ts-vad/data/alimeeting/Eval_Ali_far/rttm_dir"
script_path="/mnt/TSVAD_pytorch/wespeaker_alimeeting/external_tools/make_textgrid_rttm.py"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop through all TextGrid files in the input directory
for textgrid_file in "$input_dir"/*.TextGrid; do
    # Extract the filename without extension
    filename=$(basename "$textgrid_file" .TextGrid)
    
    # Run the Python script for each file
    python "$script_path" \
        --input_textgrid_file "$textgrid_file" \
        --output_rttm_file "$output_dir/$filename.rttm" \
        --uttid "$filename"
done
