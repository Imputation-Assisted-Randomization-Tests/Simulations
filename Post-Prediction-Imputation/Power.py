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

beta_coef = None
task_id = 1
save_file = False
max_iter = 3
L = 10000
S_size = 10

def run(Nsize, Unobserved, Single, filepath, adjust, linear_method, Missing_lambda,strata_size, small_size,verbose=1):

    # If the folder does not exist, create it
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # Create an instance of the OneShot class
    Framework = Retrain.RetrainTest(N = Nsize, covariance_adjustment=adjust)

    print("Begin")

    # Simulate data
    DataGen = Generator.DataGenerator(N = Nsize, strata_size=S_size,beta_11 = beta_coef, beta_12 = beta_coef, beta_21 = beta_coef, beta_22 = beta_coef, beta_23 = beta_coef, beta_31 = beta_coef, beta_32 = beta_coef, MaskRate=0.5,Unobserved=Unobserved, Single=Single, linear_method = linear_method,verbose=verbose,Missing_lambda = Missing_lambda)

    X, Z, U, Y, M, S = DataGen.GenerateData()

    # Flatten Z, U, Y, M, S from (50,1) to (50,)
    Z_flat = np.squeeze(Z)
    U_flat = np.squeeze(U)
    Y_flat = np.squeeze(Y)
    M_flat = np.squeeze(M)
    S_flat = np.squeeze(S)

    # Make a dataframe from X (each column separately), Z, U, Y, M, S
    df = pd.DataFrame({'X1': X[:, 0], 'X2': X[:, 1], 'X3': X[:, 2], 'X4': X[:, 3], 'X5': X[:, 4], 
                    'U': U_flat, 'Y': Y_flat,  'M': M_flat,'Z': Z_flat })

    # Print the DataFrame
    print(df.describe())
    #df.to_csv('data.csv', index=True)

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

    # Define your dictionary here based on the table you've given
    beta_to_lambda = {
        0.0: 15.338280233232549,
        0.05: 15.513632949165219,
        0.1: 15.700965399935757,
        0.15: 15.778598987947303,
        0.2: 15.919273976686219,
        0.25: 16.090606547366434,
    }

    for coef in np.arange(0.0,0.3 ,0.05):
        beta_coef = coef
        # Round to two decimal places to match dictionary keys
        beta_coef_rounded = round(beta_coef, 2)
        if beta_coef_rounded in beta_to_lambda:
            lambda_value = beta_to_lambda[beta_coef_rounded]
            run(1000, Unobserved = 1, Single = 1, filepath = "Result/HPC_power_1000_unobserved_interference" + "_single", adjust = 0, linear_method = 2,strata_size = S_size, Missing_lambda = lambda_value, small_size=False)
        else:
            print(f"No lambda value found for beta_coef: {beta_coef_rounded}")

    # Define your dictionary here based on the table you've given
    beta_to_lambda = {
        0.0: 15.64623838541569,
        0.2: 15.914767907195158,
        0.4: 16.139500824890415,
        0.6: 16.744323425885444,
        0.8: 16.996508871283982,
        1.0: 17.340156028716592,
    }

    for coef in np.arange(0.0,1.2,0.2):
        beta_coef = coef
        # Round to nearest integer to match dictionary keys
        beta_coef_rounded = round(beta_coef)
        if beta_coef_rounded in beta_to_lambda:
            lambda_value = beta_to_lambda[beta_coef_rounded]
            run(50, Unobserved = 1, Single = 1, filepath = "Result/HPC_power_50_unobserved_interference" + "_single", adjust = 0, linear_method = 2,strata_size = S_size,  Missing_lambda = lambda_value,small_size=True)
        else:
            print(f"No lambda value found for beta_coef: {beta_coef_rounded}")

"""
    for coef in np.arange(0.0,5,1):
        beta_coef = coef
        run(50, Unobserved = 1, Single = 1, filepath = "Result/HPC_power_50_unobserved_interference" + "_single", adjust = 0, linear_method = 2)

    for coef in np.arange(0,3,0.6):
        beta_coef = coef
        run(50, Unobserved = 1, Single = 1, filepath = "Result/HPC_power_50_unobserved_linearZ_linearX" + "_single", adjust = 0, linear_method = 0)

    for coef in np.arange(0.0,0.4,0.08):
        beta_coef = coef
        run(2000, Unobserved = 1, Single = 1, filepath = "Result/HPC_power_2000_unobserved_linearZ_linearX" + "_single", adjust = 0, linear_method = 0)
    
    for coef in np.arange(0.0,10,2):
        beta_coef = coef
        run(50, Unobserved = 1, Single = 1, filepath = "Result/HPC_power_50_unobserved_linearZ_nonlinearX" + "_single", adjust = 0, linear_method = 1)
    for coef in np.arange(0.0,0.80,0.16):
        beta_coef = coef
        run(2000, Unobserved = 1, Single = 1, filepath = "Result/HPC_power_2000_unobserved_linearZ_nonlinearX" + "_single", adjust = 0, linear_method = 1)
"""