#!/bin/bash
#SBATCH --account=iaifi_lab
#SBATCH --job-name=autoencoder
#SBATCH --ntasks=1 # Number of cores requested
#SBATCH --mem=100GB
#SBATCH --time=17:59:59
#SBATCH --partition=iaifi_gpu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --output=myjob.%j.out
#SBATCH --error=myjob.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pandya.sne@northeastern.edu

module load cuda/11.7.1-fasrc01
source /n/home04/spandya/miniconda3/bin/activate
conda activate gdl

python -c'import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())'
python /n/home04/spandya/DE-2-Lensing/src/scripts/train.py --config /n/home04/spandya/DE-2-Lensing/src/config/autoencoder.yaml
