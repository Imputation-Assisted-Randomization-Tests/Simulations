#!/bin/bash
#
#SBATCH --job-name=Power
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=1
#SBATCH --output=Runtime/%a.out
#SBATCH --error=Runtime/%a.err

module purge

rm -rf Runtime/*

source /scratch/jz4721/Post-prediction-Causal-Inference/venv/bin/activate
export PATH=/scratch/jz4721/Post-prediction-Causal-Inference/venv/lib64/python3.8/bin:$PATH
source ~/.bashrc

cd ../
python Power.py $SLURM_ARRAY_TASK_ID
