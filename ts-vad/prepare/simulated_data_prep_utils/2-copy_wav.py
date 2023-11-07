import os
import shutil

# Define the parent directory
parent_dir = '/home/msai/adnan002/data/simulated_data_SD'

# Get the list of directories in both 'data' and 'wav'
data_dirs = os.listdir(os.path.join(parent_dir, 'data'))
wav_dirs = os.listdir(os.path.join(parent_dir, 'wav'))

# Iterate over each directory in 'data'
for dir in data_dirs:
    # Check if the directory also exists in 'wav'
    if dir in wav_dirs:
        # Define the path to the 'wavs' directory in the 'data' directory
        wavs_dir = os.path.join(parent_dir, 'data', dir, 'wavs')

        # If the 'wavs' directory already exists, remove it
        if os.path.exists(wavs_dir):
            shutil.rmtree(wavs_dir)

        # Create the 'wavs' directory
        os.makedirs(wavs_dir, exist_ok=True)

        # Iterate over each subdirectory in the 'wav' directory
        for sub_dir in os.listdir(os.path.join(parent_dir, 'wav', dir)):
            # Iterate over each file in the subdirectory
            for file in os.listdir(os.path.join(parent_dir, 'wav', dir, sub_dir)):
                # Define the source and destination paths
                src = os.path.join(parent_dir, 'wav', dir, sub_dir, file)
                dst = os.path.join(wavs_dir, f'data_simu2_wav_{dir}_{sub_dir}_{file}')

                # Copy the file
                shutil.move(src, dst)
