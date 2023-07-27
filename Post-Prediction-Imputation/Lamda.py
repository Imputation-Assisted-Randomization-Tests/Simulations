import sys
import numpy as np
import multiprocessing
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
from sklearn.impute import SimpleImputer
import multiprocessing
import Simulation as Generator
import Retrain
import warnings
import xgboost as xgb
import os

#from cuml import XGBRegressor
 #   XGBRegressor(tree_method='gpu_hist')

beta_coef = None
task_id = 1
save_file = False
max_iter = 3
L = 2000

def run(Nsize, Unobserved, Single, filepath, adjust, linear_method):

    # If the folder does not exist, create it
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # Create an instance of the OneShot class
    Framework = Retrain.RetrainTest(N = Nsize, covariance_adjustment=adjust)

    # Simulate data
    DataGen = Generator.DataGenerator(N = Nsize, strata_size=10, beta_11 = beta_coef, beta_12 = beta_coef, beta_21 = beta_coef, beta_22 = beta_coef, beta_23 = beta_coef, beta_31 = beta_coef, beta_32 = beta_coef, MaskRate=0.5,Unobserved=Unobserved, Single=Single, linear_method = linear_method,verbose=0)
    X, Z, U, Y, M, S = DataGen.GenerateData()

def calculate_average(filename):
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    # remove any potential empty lines and convert to floats
    numbers = [float(line) for line in lines if line.strip() != ""]
        
    # calculate and return the average
    return sum(numbers) / len(numbers) if numbers else None


#print(calculate_average('lambda.txt'))
#exit()
if __name__ == '__main__':
    multiprocessing.freeze_support() # This is necessary and important, not sure why 
    # Mask Rate

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="numpy.core.getlimits")
    
    for coef in np.arange(0.0,0.3,0.05):
        if os.path.isfile("lambda.txt"):
            # If the file exists, delete it
            os.remove("lambda.txt")
        for i in range(100):
            beta_coef = coef
            run(1000, Unobserved = 1, Single = 1, filepath = "Result/HPC_power_2000_unobserved_nonlinearZ_nonlinearX" + "_single", adjust = 0, linear_method = 2)
        print("beta: "+str(coef) + "   lambda:" + str(calculate_average('lambda.txt')))
    
    print("=====================================================")
    for coef in np.arange(0.0,5,1):
        for i in range(100):
            beta_coef = coef
            run(50, Unobserved = 1, Single = 1, filepath = "Result/HPC_power_2000_unobserved_nonlinearZ_nonlinearX" + "_single", adjust = 0, linear_method = 2)
        print("beta: "+str(coef) + "   lambda:" + str(calculate_average('lambda.txt')))
    
