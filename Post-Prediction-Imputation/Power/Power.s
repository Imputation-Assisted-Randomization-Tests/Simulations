#!/bin/bash
#
#SBATCH --job-name=Power
#SBATCH --nodes=1
#SBATCH --time=01:29:00
#SBATCH --mem=1GB
#SBATCH --cpus-per-task=1
#SBATCH --output=Runtime/%a.out
#SBATCH --error=Runtime/%a.err

module purge

source /scratch/jz4721/Post-prediction-Causal-Inference/venv/bin/activate
export PATH=/scratch/jz4721/Post-prediction-Causal-Inference/venv/lib64/python3.8/bin:$PATH

cd ../
python Power.py $SLURM_ARRAY_TASK_ID
