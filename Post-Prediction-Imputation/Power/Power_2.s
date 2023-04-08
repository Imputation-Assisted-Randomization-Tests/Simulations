#!/bin/bash
#
#SBATCH --job-name=Power-2
#SBATCH --nodes=1
#SBATCH --time=0:30:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=30
#SBATCH --output=2_%a.out
#SBATCH --error=2_%a.err

module purge

source /scratch/jz4721/Post-prediction-Causal-Inference/venv/
export PATH=/scratch/jz4721/Post-prediction-Causal-Inference/venv/lib64/python3.8/bin:$PATH
source ~/.bashrc

cd ../
python3 Power.py 2 $SLURM_ARRAY_TASK_ID
