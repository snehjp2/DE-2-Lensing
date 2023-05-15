#!/bin/bash
#SBATCH --account=iaifi_lab
#SBATCH --job-name=autoencoder
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=2      # total number of tasks per node
#SBATCH --cpus-per-task=8  
#SBATCH --mem=100GB
#SBATCH --time=17:59:59
#SBATCH --partition=iaifi_gpu # Partition to submit to
#SBATCH --gres=gpu:2
#SBATCH --output=myjob.%j.out
#SBATCH --error=myjob.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pandya.sne@northeastern.edu

module purge
module load cuda/11.7.1-fasrc01
source /n/home04/spandya/miniconda3/bin/activate
conda activate gdl

export PL_TORCH_DISTRIBUTED_BACKEND=gloo

python -c'import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())'
srun python /n/home04/spandya/DE-2-Lensing/src/scripts/train.py --config /n/home04/spandya/DE-2-Lensing/src/config/autoencoder.yaml

