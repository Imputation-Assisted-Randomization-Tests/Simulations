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
L = 100

def run(Nsize, Unobserved, Single, filepath):

    # If the folder does not exist, create it
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # Create an instance of the OneShot class
    Framework = Retrain.RetrainTest(N = Nsize)

    print("Begin")

    # Simulate data
    DataGen = Generator.DataGenerator(N = Nsize, N_T = int(Nsize / 2), N_S = int(Nsize / 20), beta_11 = beta_coef, beta_12 = beta_coef, beta_21 = beta_coef, beta_22 = beta_coef, beta_23 = beta_coef, beta_31 = beta_coef, beta_32 = beta_coef, MaskRate=0.5,Unobserved=Unobserved, Single=Single)

    X, Z, U, Y, M, S = DataGen.GenerateData()

    # Oracle 
    print("Oracle")
    p_values, reject, corr_G = Framework.retrain_test(Z, X, M, Y, L=L, G = None,verbose=1)
    # Append p-values to corresponding lists
    values_oracle = [ *p_values, reject, corr_G]
    print(values_oracle)

    #Median imputer
    print("Median")
    median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    p_values, reject, corr_G = Framework.retrain_test(Z, X, M, Y,L=L, G = median_imputer,verbose=1)
    # Append p-values to corresponding lists
    values_median = [ *p_values, reject, corr_G]

    #LR imputer
    print("LR")
    BayesianRidge = IterativeImputer(estimator = linear_model.BayesianRidge(),max_iter=max_iter)
    p_values, reject, corr_G = Framework.retrain_test(Z, X, M, Y, L=L,G=BayesianRidge,verbose=1)
    # Append p-values to corresponding lists
    values_LR = [ *p_values, reject, corr_G]

    #XGBoost
    print("XGBoost")
    XGBoost = IterativeImputer(estimator = xgb.XGBRegressor(),max_iter=max_iter)
    p_values, reject, corr_G = Framework.retrain_test(Z, X, M, Y,L=L, G=XGBoost,verbose=1)
    # Append p-values to corresponding lists
    values_xgboost = [ *p_values, reject, corr_G]
    print("Finished")

    #Save the file in numpy format
    if(save_file):

        if not os.path.exists("%s/%f"%(filepath,beta_coef)):
            # If the folder does not exist, create it
            os.makedirs("%s/%f"%(filepath,beta_coef))

        # Convert lists to numpy arrays
        values_oracle = np.array(values_oracle)
        values_median = np.array(values_median)
        values_LR = np.array(values_LR)
        values_xgboost = np.array(values_xgboost)

        # Save numpy arrays to files
        np.save('%s/%f/p_values_oracle_%d.npy' % (filepath, beta_coef, task_id), values_oracle)
        np.save('%s/%f/p_values_median_%d.npy' % (filepath, beta_coef, task_id), values_median)
        np.save('%s/%f/p_values_LR_%d.npy' % (filepath, beta_coef,task_id), values_LR)
        np.save('%s/%f/p_values_xgboost_%d.npy' % (filepath, beta_coef,task_id), values_xgboost)      

if __name__ == '__main__':
    multiprocessing.freeze_support() # This is necessary and important, not sure why 
    # Mask Rate

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="numpy.core.getlimits")

    if len(sys.argv) == 2:
        task_id = int(sys.argv[1])
        save_file = True
    else:
        print("Please add the job number like this\nEx.python Power.py 1")
        exit()

    if os.path.exists("Result") == False:
        os.mkdir("Result")

    for coef in np.arange(0.01,0.5,0.05):
        beta_coef = coef
        run(1000, Unobserved = 0, Single = 1 , filepath = "Result/HPC_power_1000" + "_single")
        run(1000, Unobserved = 1, Single = 1, filepath = "Result/HPC_power_unobserved_1000" + "_single")
        run(2000, Unobserved = 1, Single = 1, filepath = "Result/HPC_power_unobserved_2000" + "_single")
        run(2000, Unobserved = 0, Single = 1 , filepath = "Result/HPC_power_2000" + "_single")
        #run(1000, Unobserved = 0, Single = 0 , filepath = "Result/HPC_power_1000" + "_multi")
        #run(1000, Unobserved = 1, Single = 0, filepath = "Result/HPC_power_unobserved_1000" + "_multi")
        #run(2000, Unobserved = 1, Single = 0, filepath = "Result/HPC_power_unobserved_2000" + "_multi")
        #run(2000, Unobserved = 0, Single = 0 , filepath = "Result/HPC_power_2000" + "_multi")
