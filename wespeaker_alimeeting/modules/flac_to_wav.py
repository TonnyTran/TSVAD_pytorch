from pydub import AudioSegment
import os
import glob, tqdm, os, textgrid, soundfile, copy, json, argparse, numpy, torch, wave
from collections import defaultdict
from speaker_encoder import ECAPA_TDNN

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_directory', help='the path for the dihard data files')
    args = parser.parse_args()
    args.output_directory = os.path.join(args.data_directory, 'wav')
    args.input_directory = os.path.join(args.data_directory, 'flac')
    return args

def convert_flac_to_wav(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through FLAC files in the input directory
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".flac"):
                flac_path = os.path.join(root, file)
                wav_path = os.path.join(output_dir, os.path.splitext(file)[0] + ".wav")

                # Load FLAC file and export as WAV
                audio = AudioSegment.from_file(flac_path, format="flac")
                audio.export(wav_path, format="wav")

def main():
    args = get_args()

    input_directory = args.input_directory 
    output_directory = args.output_directory

    convert_flac_to_wav(input_directory, output_directory)

if __name__ == '__main__':
	main()