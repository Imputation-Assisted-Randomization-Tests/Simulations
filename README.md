# Post-prediction-Causal-Inference
In many practical studies, among the outcomes Y1, . . . , YK , one or multiple of them are
incomplete. In randomized experiments, simply ignoring the missing outcomes may lead to
statistical inference not finite population exact anymore. In matched observational studies,
people routinely exclude study subjects with missing outcomes, which would substantially
the statistical power due to discarding many study samples.
These framework is for testing post-prediction missing data imputation


python >= 3.8
python -m pip install --upgrade pip
pip -version >= 22.0.4


# For HPC Users

### Prepare the environment
- please go to your scratch palce
  `cd /scratch/$YourNetID`
- Get the repository 
  `git clone https://github.com/jiawei-zhang-a/Post-prediction-Causal-Inference.git`
- Get into the directory
 `cd Post-prediction-Causal-Inference`
 - Get the python module from HPC to have a python version great than 3.8
`module load  python/intel/3.8.6`
 - Create the venv 
 `python -m venv venv`
 - Upgrade the pip vesion
`pip install --upgrade pip`
- Install all the requirements
  `pip install -r requirements.txt`

###
Get started
- Get into the directory to start
  `cd /Post-Prediction-Imputation/Power` 
- make a folder for runtime files collections
  `mkdir Runtime`
- Change the sbatch file to fit your local env by change the YourNetID
  `source /scratch/YourNetID/Post-prediction-Causal-Inference/venv/bin/activate`
  `export PATH=/scratch/YourNetID/Post-prediction-Causal-Inference/venv/lib64/python3.8/bin:$PATH`
- submit array jobs
  `sbatch --array=1-2000 Power.s`
- Check your jobs
 `squeue -u $USER`

The method used to semi-supervised learning for the imputation is 

KNN,Missforest,XGBoost,NN,RidgeRegression,Kernel Approximation,Mean and median
