#!/bin/bash
#SBATCH --partition=SCSEGPU_M1 
#SBATCH --qos=q_amsai 
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1 
#SBATCH --mem=1G 
#SBATCH --job-name=preparesimtrain
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

# Define the list of elements
elements=('swb_sre_cv_ns1_beta2_200' 'swb_sre_cv_ns4_beta9_200' 'swb_sre_tr_ns2_beta2_1000' 'swb_sre_cv_ns3_beta5_200' 'swb_sre_tr_ns1_beta2_1000' 'swb_sre_tr_ns3_beta5_1000' 'swb_sre_cv_ns2_beta2_200' 'swb_sre_tr_ns4_beta9_1000')


# Iterate over each element in the list
for element in "${elements[@]}"
do
    # Run the python command with the current element
    python /home/msai/adnan002/repos/TSVAD_pytorch/ts-vad/prepare/prepare_simulated.py \
        --data_path "/home/msai/adnan002/data/simulated_data_SD/data/$element" \
        --type simtrain \
        --source /home/msai/adnan002/repos/TSVAD_pytorch/ts-vad/models/ecapa-tdnn.model
done