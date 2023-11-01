import os, wave
import tqdm, glob

directory = "/home/users/ntu/tlkushag/scratch/data08/dihard/third_dihard_challenge_eval/data/target_audio"

def int2str(num, length):
    num_str = str(num)
    while len(num_str) < length:
        num_str = '0' + num_str
    return num_str

def print_zero_file_path():
    spkcnt = 0
    allcnt = 0
    dirwithmore4files = 0
    for i in range (1, 260):
        subdirectory = 'DH_EVAL_' + int2str(i, 4)
        dir = os.path.join(directory, subdirectory)
        files = sorted(glob.glob(dir + "/*.wav"))
        if(len(files)>5):
            dirwithmore4files += 1
            # print("Directory: ", dir)
            # print("Length: ", len(files))
            # print("Files: ", files)

        for file in files:
            print(file)
            if 'all.wav' in file:
                allcnt += 1
            else:
                spkcnt += 1
            # wav_length = wave.open(file, 'rb').getnframes()
            # # if (wav_length - int(6 * 16000)) < 0:
            # if(wav_length==0):
            #     print("File path: ", os.path.join(dir, file))
            #     # print("Length: ", wav_length)
                # print("Duration: ", wav_length / 16000)
    print("All count: ", allcnt)
    print("Speaker count: ", spkcnt)
    print("Directory with more files: ", dirwithmore4files)
# print_zero_file_path()

filepath = '/home/users/ntu/tlkushag/scratch/data08/dihard/third_dihard_challenge_eval/data/target_audio/DH_EVAL_0002/all.wav'
import wave

def get_wav_duration(filepath):
    with wave.open(filepath, 'r') as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration
# print(get_wav_duration(filepath))


def sum_textgrid_sizes(folder_path):
    total_size = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.TextGrid'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if 'size' in line:
                        chunks = line.split()
                        num_size = chunks[-1]
                        num_size = int(num_size)
                        if num_size > 4:
                            total_size += 4
                        else:
                            total_size += (int(num_size))
                        break
    return total_size

# folder_path = '/home/users/ntu/tlkushag/scratch/data08/dihard/third_dihard_challenge_eval/data/textgrid'
# print(sum_textgrid_sizes(folder_path))


import os

def delete_line_with_string(filepath, string):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    with open(filepath, 'w') as file:
        for line in lines:
            if string not in line:
                file.write(line)
    return os.path.isfile(filepath)

# filepath = "/home/users/ntu/tlkushag/scratch/TSVAD_pytorch/ts-vad/40epochs copy.o3333505"
# string = '/646'
# print(delete_line_with_string(filepath, string))
# string = 'Train'
# print(delete_line_with_string(filepath, string))
# string = 'Eval'
# print(delete_line_with_string(filepath, string))


def filterFiles(directory):
    print(f"For Directory: {directory}")
    if 'third_dihard_challenge_eval' in directory:
        typ = 'eval'
        partsubdir = 'DH_EVAL_'
    if 'third_dihard_challenge_dev' in directory:
        typ = 'dev'
        partsubdir = 'DH_DEV_'
    total_ignored_duration = 0
    total_duration = 0
    for i in range (1, 260):
        subdirectory = partsubdir + int2str(i, 4)
        dir = os.path.join(directory, subdirectory)
        files = sorted(glob.glob(dir + "/*.wav"))
        print(f"[{subdirectory}]")
        total_speech_duration = 0
        for file in files:
            if 'all.wav' not in file:
                duration = get_wav_duration(file)
                total_speech_duration += duration
        print(f"Total speech duration: {total_speech_duration}")
        total_duration += total_speech_duration
        
        if(len(files)<6):
            continue
        # get the duration of the 5th file in the directory
        sum_duration = 0
        for file in files:
            # if file index is less than 4  then continue
            if files.index(file) < 4 or 'all.wav' in file:
                continue
            duration = get_wav_duration(file)
            sum_duration += duration
        total_ignored_duration += sum_duration
        print(f"Total Ignored Duration: {sum_duration}")
    print("-------------------------------------------")
    print(f"Total Ignored Duration inside {typ}: {total_ignored_duration}")
    print(f"Total Duration inside {typ}: {total_duration}")
    print(f"Percentage Ignored inside {typ}: {(total_ignored_duration/total_duration)*100}")
    print("-------------------------------------------")


eval_dir = "/home/users/ntu/tlkushag/scratch/data08/dihard/third_dihard_challenge_eval/data/target_audio"
dev_dir = "/home/users/ntu/tlkushag/scratch/data08/dihard/third_dihard_challenge_dev/data/target_audio"
filterFiles(eval_dir)
filterFiles(dev_dir)
