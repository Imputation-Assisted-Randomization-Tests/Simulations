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

def run(Nsize, Unobserved, Single, adjust, strata_size):

    # Create an instance of the OneShot class
    Framework = Retrain.RetrainTest(N = Nsize, covariance_adjustment=adjust)

    # Simulate data
    DataGen = Generator.DataGenerator(N = Nsize, strata_size=strata_size, beta_11 = beta_coef, beta_12 = beta_coef, beta_21 = beta_coef, beta_22 = beta_coef, beta_23 = beta_coef, beta_31 = beta_coef, beta_32 = beta_coef, MaskRate=0.5,Unobserved=Unobserved, Single=Single,verbose=0)
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
    # Mask Rate

    for coef in np.arange(0.0, 0.18, 0.03):
        if os.path.isfile("lambda1.txt"):
            # If the file exists, delete it
            os.remove("lambda1.txt")
        if os.path.isfile("lambda2.txt"):
            # If the file exists, delete it
            os.remove("lambda2.txt")
        if os.path.isfile("lambda3.txt"):
            # If the file exists, delete it
            os.remove("lambda3.txt")
        for i in range(100):
            beta_coef = coef
            run(1000, Unobserved = 1, Single = 0, strata_size = 10, adjust = 0)
        avg_lambda1 = calculate_average('lambda1.txt')
        print("beta: "+str(coef) + "   lambda1:" + str(avg_lambda1))
        avg_lambda2 = calculate_average('lambda2.txt')
        print("beta: "+str(coef) + "   lambda2:" + str(avg_lambda2))
        avg_lambda3 = calculate_average('lambda3.txt')
        print("beta: "+str(coef) + "   lambda3:" + str(avg_lambda3))
    print("=====================================================")

    for coef in np.arange(0.0, 0.72, 0.12):
        if os.path.isfile("lambda1.txt"):
            # If the file exists, delete it
            os.remove("lambda1.txt")
        if os.path.isfile("lambda2.txt"):
            # If the file exists, delete it
            os.remove("lambda2.txt")
        if os.path.isfile("lambda3.txt"):
            # If the file exists, delete it
            os.remove("lambda3.txt")
        for i in range(100):
            beta_coef = coef
            run(50, Unobserved = 1, Single = 0,  strata_size = 10,adjust = 0)
        avg_lambda1 = calculate_average('lambda1.txt')
        print("beta: "+str(coef) + "   lambda1:" + str(avg_lambda1))
        avg_lambda2 = calculate_average('lambda2.txt')
        print("beta: "+str(coef) + "   lambda2:" + str(avg_lambda2))
        avg_lambda3 = calculate_average('lambda3.txt')
        print("beta: "+str(coef) + "   lambda3:" + str(avg_lambda3))
    print("=====================================================")


