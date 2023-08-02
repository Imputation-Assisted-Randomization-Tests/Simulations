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
from catboost import CatBoostRegressor

beta_coef = None
task_id = 1
save_file = False
max_iter = 3
L = 100
S_size = 10

class CatBoostRegressorForImputer(CatBoostRegressor):
    def transform(self, X):
        return self.predict(X)

def run(Nsize, Unobserved, Single, filepath, adjust, linear_method, strata_size, Missing_lambda = None,verbose=1):

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
    #print(df.describe())
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
    print("XGBoost")
    XGBoost = IterativeImputer(estimator=xgb.XGBRegressor(n_jobs=1), max_iter=max_iter)
    p_values, reject, test_time = Framework.retrain_test(Z, X, M, Y, strata_size = strata_size,L=L, G=XGBoost, verbose=1)
    values_xgboost = [*p_values, reject, test_time]

    #CatBoost
    """print("CatBoost")
    catboost = IterativeImputer(estimator=CatBoostRegressorForImputer(verbose=0,thread_count=1), max_iter=max_iter)
    p_values, reject, test_time = Framework.retrain_test(Z, X, M, Y, strata_size = strata_size,L=L, G=catboost, verbose=1)
    values_catboost = [*p_values, reject, test_time] """

    #LightGBM
    print("LightGBM")
    #start_time = time.time()
    LightGBM = IterativeImputer(estimator=lgb.LGBMRegressor(n_jobs=1), max_iter=max_iter)
    p_values, reject, test_time = Framework.retrain_test(Z, X, M, Y, strata_size=strata_size,L=L, G=LightGBM, verbose=verbose)
    #end_time = time.time()
    values_lightgbm = [*p_values, reject, test_time]
    #print(f"Execution time for LightGBM: {end_time - start_time} seconds\n")

    #Save the file in numpy format
    if(save_file):

        if not os.path.exists("%s/%f"%(filepath,beta_coef)):
            # If the folder does not exist, create it
            os.makedirs("%s/%f"%(filepath,beta_coef))

        # Convert lists to numpy arrays
        values_median = np.array(values_median)
        values_LR = np.array(values_LR)
        values_lightgbm = np.array(values_lightgbm)
        values_oracle = np.array(values_oracle)
        values_xgboost = np.array(values_xgboost)
        values_catboost = np.array(values_catboost)

        # Save numpy arrays to files
        np.save('%s/%f/p_values_oracle_%d.npy' % (filepath, beta_coef, task_id), values_oracle)
        np.save('%s/%f/p_values_median_%d.npy' % (filepath, beta_coef, task_id), values_median)
        np.save('%s/%f/p_values_LR_%d.npy' % (filepath, beta_coef,task_id), values_LR)
        np.save('%s/%f/p_values_lightGBM_%d.npy' % (filepath, beta_coef, task_id), values_lightgbm)
        np.save('%s/%f/p_values_xgboost_%d.npy' % (filepath, beta_coef, task_id), values_xgboost)
        np.save('%s/%f/p_values_catboost_%d.npy' % (filepath, beta_coef, task_id), values_catboost)

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
        run(100, Unobserved = 1, Single = 1, filepath = "Result/HPC_power_100_unobserved_linearZ_linearX" + "_single", adjust = 0, linear_method = 0,strata_size = S_size)
    
    for coef in np.arange(0.0,0.4,0.08):
        beta_coef = coef
        run(1000, Unobserved = 1, Single = 1, filepath = "Result/HPC_power_1000_unobserved_linearZ_linearX" + "_single", adjust = 0, linear_method = 0,strata_size = S_size)
    
    for coef in np.arange(0.0,5,1):
        beta_coef = coef
        run(100, Unobserved = 1, Single = 1, filepath = "Result/HPC_power_100_unobserved_linearZ_nonlinearX" + "_single", adjust = 0, linear_method = 1,strata_size = S_size)
    
    for coef in np.arange(0.0,0.80,0.16):
        beta_coef = coef
        run(1000, Unobserved = 1, Single = 1, filepath = "Result/HPC_power_1000_unobserved_linearZ_nonlinearX" + "_single", adjust = 0, linear_method = 1,strata_size = S_size)

    for coef in np.arange(0.0,0.3 ,0.05):
        beta_coef = coef
        run(1000, Unobserved = 1, Single = 1, filepath = "Result/HPC_power_1000_unobserved_nonlinearZ_nonlinearX" + "_single", adjust = 0, linear_method = 2,strata_size = S_size)
    for coef in np.arange(0.0,1.5,0.25):
        beta_coef = coef
        run(100, Unobserved = 1, Single = 1, filepath = "Result/HPC_power_100_unobserved_nonlinearZ_nonlinearX" + "_single", adjust = 0, linear_method = 2,strata_size = S_size)
