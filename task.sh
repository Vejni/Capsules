#!/bin/bash

#SBATCH --partition=uoa-gpu
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.veiner.17@abdn.ac.uk
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cgpu01

date 
hostname

module load python-3.7.7
module load anaconda3
module list

source /uoa/home/u02mv17/Repository/.venv/bin/activate

srun python /uoa/home/u02mv17/Repository/main.py

date
exit 0
