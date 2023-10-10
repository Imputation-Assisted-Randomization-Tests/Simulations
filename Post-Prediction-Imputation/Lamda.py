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

def run(Nsize,  Single, adjust, strata_size, linear_method):


    # Create an instance of the OneShot class
    Framework = Retrain.RetrainTest(N = Nsize, covariance_adjustment=adjust)

    # Simulate data
    DataGen = Generator.DataGenerator(N = Nsize, strata_size=strata_size, beta_11 = beta_coef, beta_12 = beta_coef, beta_21 = beta_coef, beta_22 = beta_coef, beta_23 = beta_coef, beta_31 = beta_coef, beta_32 = beta_coef, MaskRate=0.5,Single=Single, linear_method = linear_method,verbose=0)
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
    beta_to_lambda = {}

    coef_ranges = [(0,1.5,0.25), (0.0,0.42,0.07), (0.0,4.8,0.8), (0.0,0.96,0.16), (0.0,1.5,0.25), (0.0,0.36,0.06)]
    filepath_prefixes = ["Result/HPC_power_50_unobserved_linearZ_linearX", "Result/HPC_power_1000_unobserved_linearZ_linearX", 
                        "Result/HPC_power_50_unobserved_linearZ_nonlinearX", "Result/HPC_power_1000_unobserved_linearZ_nonlinearX", 
                        "Result/HPC_power_50_unobserved_nonlinearZ_nonlinearX", "Result/HPC_power_1000_unobserved_nonlinearZ_nonlinearX"]
    linear_methods = [0, 0, 1, 1, 2, 2]
    small_sizes = [True, False, True, False, True, False]

    for coef_range,  linear_method, small_size in zip(coef_ranges,  linear_methods, small_sizes):
        for coef in np.arange(*coef_range):
            beta_coef = coef
            if os.path.isfile("lambda.txt"):
                # If the file exists, delete it
                os.remove("lambda.txt")
            for i in range(100):
                if small_size:
                    run(1000, Single = 1,  strata_size = 10, adjust = 0, linear_method = linear_method)
                else:
                    run(1000,  Single = 1,  strata_size = 10, adjust = 0, linear_method = linear_method)
            avg_lambda = calculate_average('lambda.txt')
            print("beta: " + str(coef) + "   lambda:" + str(avg_lambda))

    print("=====================================================")
