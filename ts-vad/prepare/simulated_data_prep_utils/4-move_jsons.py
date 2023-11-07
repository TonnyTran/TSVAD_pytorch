import os

# Define the base path
base_path = "/home/msai/adnan002/data/simulated_data_SD/data"

# Define the target directory
target_dir = os.path.join(base_path, "all_files")

# Define the target file
target_file = os.path.join(target_dir, "all_simtrain.json")

# Initialize an empty string to hold all data
all_data = ""

# Iterate over all directories in the base path
for dir_name in os.listdir(base_path):
    dir_path = os.path.join(base_path, dir_name)
    
    # Skip the target directory
    if dir_path == target_dir:
        continue
    
    # Define the path to the ts_simtrain.json file in this directory
    file_path = os.path.join(dir_path, "ts_simtrain.json")
    
    # If the file exists, read it and add its data to all_data
    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            data = f.read()
            all_data += data + "\n"

# Write all_data to the target file
with open(target_file, 'w') as f:
    f.write(all_data)
