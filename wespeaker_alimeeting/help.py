import os, wave
import tqdm, glob

directory = "/home/users/ntu/tlkushag/scratch/data08/dihard/third_dihard_challenge_eval/data/target_audio"

def int2str(num, length):
    num_str = str(num)
    while len(num_str) < length:
        num_str = '0' + num_str
    return num_str

def print_zero_file_path():
    for i in range (1, 250):
        subdirectory = 'DH_EVAL_' + int2str(i, 4)
        dir = os.path.join(directory, subdirectory)
        files = sorted(glob.glob(dir + "/*.wav"))
        for file in files:
            wav_length = wave.open(file, 'rb').getnframes()
            if (wav_length - int(6 * 16000)) < 0:
                print("File path: ", os.path.join(dir, file))
                print("Length: ", wav_length)
                print("Duration: ", wav_length / 16000)

filepath = '/home/users/ntu/tlkushag/scratch/data08/dihard/third_dihard_challenge_eval/data/target_audio/DH_EVAL_0002/3.wav'

import soundfile as sf
def get_wav_duration(filepath):
    audio, samplerate = sf.read(filepath)
    print(f"Audio file: {filepath}")
    print(f"Audio shape: {audio.shape}")
    print(f"Samplerate: {samplerate}")
    duration = len(audio) / samplerate
    print(f"Duration (s): {duration}")
    return int(duration * 1000)

# print_zero_file_path()

import os

def delete_line_with_string(filepath, string):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    with open(filepath, 'w') as file:
        for line in lines:
            if string not in line:
                file.write(line)
    return os.path.isfile(filepath)

# filepath = '/home/users/ntu/tlkushag/scratch/TSVAD_pytorch/ts-vad/z.o3103949'
# string = 'filename'
# print(delete_line_with_string(filepath, string))

