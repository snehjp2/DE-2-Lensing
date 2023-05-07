#!/bin/bash


setup_str=$( python distributed_data_parallel_slurm_setup.py "$@" )
eval $setup_str


module load cuda/11.7.1-fasrc01
source /n/home04/spandya/miniconda3/bin/activate
conda activate gdl

python ddp_train.py
