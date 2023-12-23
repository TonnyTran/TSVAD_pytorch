import os
import sys
import glob

# Get the directory path from the command-line arguments
# dir_path = "/home/msai/adnan002/data/simulated_data_SD/data"
dir_path = "data/v2_simulated_data_Switchboard_SRE_small_16k/data/simu3/data"

# Find all 'rttm' files in the directory and its subdirectories
rttm_files = glob.glob(os.path.join(dir_path, '**', 'rttm'), recursive=True)

# Process each file
for input_file_path in rttm_files:
    # Get the directory of the input file
    input_dir = os.path.dirname(input_file_path)

    # Create the output directory path
    output_dir = os.path.join(input_dir, 'rttms')

    # delete output directory if it already exists
    if os.path.exists(output_dir):
        print (f'Removing {output_dir}')
        os.system(f'rm -rf {output_dir}')

    os.makedirs(output_dir)

    # Open the input RTTM file
    with open(input_file_path, 'r') as infile:
        for line in infile:
            # Split the line into fields
            fields = line.split()
            
            # Get the speaker ID (second field)
            speaker_id = fields[1]
            
            # Open the corresponding output file and append the line to it
            with open(os.path.join(output_dir, f'{speaker_id}.rttm'), 'a') as outfile:
                outfile.write(line)
