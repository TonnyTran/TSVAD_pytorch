#!/bin/bash
#PBS -q normal
#PBS -j oe
#PBS -l ngpus=1
#PBS -l mem=20gb
#PBS -N tsvadtrain_8
#PBS -l walltime=06:00:00
#PBS -P Personal
source /home/users/ntu/adnan002/scratch/miniconda3/bin/activate wespeak2
cd /home/users/ntu/adnan002/scratch/release/TSVAD_pytorch/ts-vad
./run_train.sh