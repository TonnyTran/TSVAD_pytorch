# Define the list of elements
# Add folders in elements which needs to be prepared
elements=('swb_sre_cv_ns3_beta5_200' 'swb_sre_cv_ns4_beta9_200' 'swb_sre_cv_ns5_beta14_200' 'swb_sre_cv_ns6_beta20_200' 'swb_sre_tr_ns1_beta2_1000' 'swb_sre_tr_ns2_beta2_1000' 'swb_sre_tr_ns3_beta5_1000' 'swb_sre_tr_ns4_beta9_1000' 'swb_sre_tr_ns5_beta14_1000' 'swb_sre_tr_ns6_beta20_1000')

# Iterate over each element in the list
for element in "${elements[@]}"
do
    # Run the python command with the current element
    python prepare_simulated.py \
        --data_path "data/v2_simulated_data_Switchboard_SRE_small_16k/data/simu3/data/$element" \
        --type simtrain \
        --source ts-vad/pretrained_models/ecapa-tdnn.model
done