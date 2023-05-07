#!/bin/bash -l
#SBATCH --account=iaifi_lab
#SBATCH --job-name=autoencoder
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=100GB
#SBATCH --time=17:59:59
#SBATCH --partition=iaifi_gpu # Partition to submit to
#SBATCH --gres=gpu:4
#SBATCH --output=myjob.%j.out
#SBATCH --error=myjob.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pandya.sne@northeastern.edu

master_addr=$(hostname -i)
master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')


# --- Call the Script which the User will Edit ---
# With srun this will be run on all nodes for all processes
# activate conda env and relevant modules before running
srun --gres=gpu:1 ./train.bash --master_addr $master_addr --master_port $master_port
