#!/bin/bash --login

#SBATCH --partition=uoa-gpu
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.veiner.17@abdn.ac.uk
#SBATCH --nodes=1
#SBATCH --ntasks=1


date 
hostname

module load cudatoolkit-10.1.168
module list

source activate pytorch
source /uoa/home/u02mv17/Repository/.venv/bin/activate

nvidia-smi 
python /uoa/home/u02mv17/Repository/main.py

date
exit 0
