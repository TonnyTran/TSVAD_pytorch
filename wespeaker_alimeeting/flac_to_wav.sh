#!/bin/bash

input_directory="/home/users/ntu/tlkushag/scratch/data08/dihard/third_dihard_challenge_eval/data/flac"  # Replace with the path to your input directory
output_directory="/home/users/ntu/tlkushag/scratch/data08/dihard/third_dihard_challenge_eval/data/wav"  # Replace with the path to your output directory

# Create the output directory if it doesn't exist
mkdir -p "$output_directory"

# Loop through each FLAC file in the input directory
for flac_file in "$input_directory"/*.flac; do
    if [ -f "$flac_file" ]; then
        # Get the base filename without extension
        base_filename=$(basename -- "$flac_file")
        filename_no_extension="${base_filename%.*}"
        
        # Define the output WAV file path
        wav_file="$output_directory/$filename_no_extension.wav"
        
        # Use FFmpeg to convert the FLAC file to WAV
        ffmpeg -i "$flac_file" "$wav_file"
        
        echo "Converted: $flac_file to $wav_file"
    fi
done

echo "Conversion complete."
