#!/bin/bash

# Define paths
rttm_dir="/home/users/ntu/tlkushag/scratch/data08/dihard/third_dihard_challenge_eval/data/rttm"      # Replace with the path to your RTTM files
output_textgrid_dir="/home/users/ntu/tlkushag/scratch/data08/dihard/third_dihard_challenge_eval/data/textgrid"  # Replace with the path for TextGrid output
curr_path="/home/users/ntu/tlkushag/scratch/TSVAD_pytorch/wespeaker_alimeeting"
# Create the output directory if it doesn't exist
mkdir -p "$output_textgrid_dir"

# Loop through all RTTM files in the subdirectory
for rttm_file in "$rttm_dir"/*.rttm; do
    # Extract the file name without extension
    file_name=$(basename "$rttm_file" .rttm)
    
    # Define the output TextGrid file path
    output_textgrid_file="$output_textgrid_dir/${file_name}.TextGrid"

    # Call the rttm_to_textgrid.py script on the current RTTM file
    python ${curr_path}/modules/rttm_to_textgrid.py \
    --input_rttm_file "$rttm_file" \
    --output_textgrid_file "$output_textgrid_file"
done

echo "Conversion complete. TextGrid files are saved in $output_textgrid_dir."
