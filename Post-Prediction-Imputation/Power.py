import sys
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
from sklearn.impute import SimpleImputer
import Simulation as Generator
import Retrain
import os
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
#from catboost import CatBoostRegressor

beta_coef = None
task_id = 1
save_file = False
max_iter = 3
L = 100
S_size = 10

#class CatBoostRegressorForImputer(CatBoostRegressor):
#    def transform(self, X):
#        return self.predict(X)

def run(Nsize, Unobserved, Single, filepath, adjust, linear_method, strata_size,small_size, Missing_lambda = None,verbose=1):

    # If the folder does not exist, create it
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # Create an instance of the OneShot class
    Framework = Retrain.RetrainTest(N = Nsize, covariance_adjustment=adjust)

    print("Begin")

    # Simulate data
    DataGen = Generator.DataGenerator(N = Nsize, strata_size=S_size,beta_11 = beta_coef, beta_12 = beta_coef, beta_21 = beta_coef, beta_22 = beta_coef, beta_23 = beta_coef, beta_31 = beta_coef, beta_32 = beta_coef, MaskRate=0.5,Unobserved=Unobserved, Single=Single, linear_method = linear_method,verbose=verbose,Missing_lambda = Missing_lambda)

    X, Z, U, Y, M, S = DataGen.GenerateData()

    #Oracale imputer
    print("Oracle")
    p_values, reject, test_time = Framework.retrain_test(Z, X, M, Y,strata_size = strata_size, L=L, G = None,verbose=0)
    # Append p-values to corresponding lists
    values_oracle = [ *p_values, reject, test_time]

    #Median imputer
    print("Median")
    median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    p_values, reject, test_time = Framework.retrain_test(Z, X, M, Y, strata_size = strata_size,L=L, G = median_imputer,verbose=verbose)
    # Append p-values to corresponding lists
    values_median = [ *p_values, reject, test_time]

    #LR imputer
    print("LR")
    BayesianRidge = IterativeImputer(estimator = linear_model.BayesianRidge(),max_iter=max_iter)
    p_values, reject, test_time = Framework.retrain_test(Z, X, M, Y,strata_size=strata_size, L=L,G=BayesianRidge,verbose=verbose)
    # Append p-values to corresponding lists
    values_LR = [ *p_values, reject, test_time]

    #XGBoost
    if small_size == True:
        XGBoost = IterativeImputer(estimator=xgb.XGBRegressor(n_jobs=1), max_iter=max_iter)
        p_values, reject, test_time = Framework.retrain_test(Z, X, M, Y, strata_size = strata_size,L=L, G=XGBoost, verbose=1)
        values_xgboost = [*p_values, reject, test_time]

    #LightGBM
    if small_size == False:
        print("LightGBM")
        LightGBM = IterativeImputer(estimator=lgb.LGBMRegressor(n_jobs=1), max_iter=max_iter)
        p_values, reject, test_time = Framework.retrain_test(Z, X, M, Y, strata_size=strata_size,L=L, G=LightGBM, verbose=verbose)
        values_lightgbm = [*p_values, reject, test_time]

    #Save the file in numpy format
    if(save_file):

        if not os.path.exists("%s/%f"%(filepath,beta_coef)):
            # If the folder does not exist, create it
            os.makedirs("%s/%f"%(filepath,beta_coef))

        # Save numpy arrays to files
        np.save('%s/%f/p_values_oracle_%d.npy' % (filepath, beta_coef, task_id), values_oracle)
        np.save('%s/%f/p_values_median_%d.npy' % (filepath, beta_coef, task_id), values_median)
        np.save('%s/%f/p_values_LR_%d.npy' % (filepath, beta_coef,task_id), values_LR)
        if small_size == False:
            np.save('%s/%f/p_values_lightGBM_%d.npy' % (filepath, beta_coef, task_id), values_lightgbm)
        if small_size == True:
            np.save('%s/%f/p_values_xgboost_%d.npy' % (filepath, beta_coef, task_id), values_xgboost)

if __name__ == '__main__':

    if len(sys.argv) == 2:
        task_id = int(sys.argv[1])
        save_file = True
    else:
        print("Please add the job number like this\nEx.python Power.py 1")
        exit()

    if os.path.exists("Result") == False:
        os.mkdir("Result")

    for coef in np.arange(0,1.5,0.3):
        beta_coef = coef
        run(50, Unobserved = 1, Single = 1, filepath = "Result/HPC_power_50_unobserved_linearZ_linearX" + "_single", adjust = 0, linear_method = 0,strata_size = S_size, small_size = True)
    for coef in np.arange(0.0,0.4,0.08):
        beta_coef = coef
        run(1000, Unobserved = 1, Single = 1, filepath = "Result/HPC_power_1000_unobserved_linearZ_linearX" + "_single", adjust = 0, linear_method = 0,strata_size = S_size, small_size = False)
    
    for coef in np.arange(0.0,5,1):
        beta_coef = coef
        run(50, Unobserved = 1, Single = 1, filepath = "Result/HPC_power_50_unobserved_linearZ_nonlinearX" + "_single", adjust = 0, linear_method = 1,strata_size = S_size, small_size = True)
    for coef in np.arange(0.0,0.80,0.16):
        beta_coef = coef
        run(1000, Unobserved = 1, Single = 1, filepath = "Result/HPC_power_1000_unobserved_linearZ_nonlinearX" + "_single", adjust = 0, linear_method = 1,strata_size = S_size, small_size = False)

    for coef in np.arange(0.0,1.5,0.25):
        beta_coef = coef
        run(50, Unobserved = 1, Single = 1, filepath = "Result/HPC_power_50_unobserved_nonlinearZ_nonlinearX" + "_single", adjust = 0, linear_method = 2,strata_size = S_size, small_size = True)
    for coef in np.arange(0.0,0.3 ,0.05):
        beta_coef = coef
        run(1000, Unobserved = 1, Single = 1, filepath = "Result/HPC_power_1000_unobserved_nonlinearZ_nonlinearX" + "_single", adjust = 0, linear_method = 2,strata_size = S_size, small_size = False)
