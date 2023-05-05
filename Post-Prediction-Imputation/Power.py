import sys
import numpy as np
import multiprocessing
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
from sklearn.impute import SimpleImputer
import multiprocessing
import Simulation as Generator
import OneShot
import warnings
import xgboost as xgb
import os

#from cuml import XGBRegressor
 #   XGBRegressor(tree_method='gpu_hist')

beta_coef = None
task_id = 1
save_file = False
max_iter = 10

def run(Nsize, Unobserved, Single, filepath):

    # If the folder does not exist, create it
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # Create an instance of the OneShot class
    Framework = OneShot.OneShotTest(N = Nsize, Single=Single)

    print("Begin")

    # Simulate data
    DataGen = Generator.DataGenerator(N = Nsize, N_T = int(Nsize / 2), N_S = int(Nsize / 20), beta_11 = beta_coef, beta_12 = beta_coef, beta_21 = beta_coef, beta_22 = beta_coef, beta_23 = beta_coef, beta_31 = beta_coef, beta_32 = beta_coef, MaskRate=0.5,Unobserved=Unobserved, Single=Single)

    X, Z, U, Y, M, S = DataGen.GenerateData()

    print(X.shape, Z.shape, U.shape, Y.shape)

    # Oracle 
    p11, p12, p21, p22, p31, p32, corr1, corr2, reject = Framework.one_shot_test(Z, X, M, Y, L=5000, G1=None, G2=None,verbose=0)
    # Append p-values to corresponding lists
    if Single:
        p_values_oracle = [ p11, p12, p21, p22, p31, p32, corr1[0], corr2[0],reject ]
    else:
        p_values_oracle = [ p11, p12, p21, p22, p31, p32, corr1[2], corr2[2],reject ]
    
    #Median imputer
    median_imputer_1 = SimpleImputer(missing_values=np.nan, strategy='median')
    median_imputer_2 = SimpleImputer(missing_values=np.nan, strategy='median')
    p11, p12, p21, p22, p31, p32, corr1, corr2, reject = Framework.one_shot_test(Z, X, M, Y,L=5000, G1=median_imputer_1, G2=median_imputer_2,verbose=0)
    # Append p-values to corresponding lists
    if Single:
        p_values_median = [ p11, p12, p21, p22, p31, p32, corr1[0], corr2[0],reject ]
    else:
        p_values_median = [ p11, p12, p21, p22, p31, p32, corr1[2], corr2[2],reject ]

    #LR imputer
    BayesianRidge_1 = IterativeImputer(estimator = linear_model.BayesianRidge(),max_iter=max_iter)
    BayesianRidge_2 = IterativeImputer(estimator = linear_model.BayesianRidge(),max_iter=max_iter)
    p11, p12, p21, p22, p31, p32, corr1, corr2, reject = Framework.one_shot_test(Z, X, M, Y, L=5000,G1=BayesianRidge_1, G2=BayesianRidge_2,verbose=0)
    # Append p-values to corresponding lists
    if Single:
        p_values_LR = [ p11, p12, p21, p22, p31, p32, corr1[0], corr2[0],reject ]
    else:
        p_values_LR = [ p11, p12, p21, p22, p31, p32, corr1[2], corr2[2],reject ]
        
    #XGBoost
    XGBoost_1= IterativeImputer(estimator = xgb.XGBRegressor(),max_iter=max_iter)
    XGBoost_2= IterativeImputer(estimator = xgb.XGBRegressor(),max_iter=max_iter)
    p11, p12, p21, p22, p31, p32, corr1, corr2, reject = Framework.one_shot_test(Z, X, M, Y,L=5000, G1=XGBoost_1, G2=XGBoost_2,verbose=0)
    # Append p-values to corresponding lists
    if Single:
        p_values_xgboost = [ p11, p12, p21, p22, p31, p32, corr1[0], corr2[0],reject ]
    else:
        p_values_xgboost = [ p11, p12, p21, p22, p31, p32, corr1[2], corr2[2],reject ]
    print("Finished")

    #Save the file in numpy format
    if(save_file):

        if not os.path.exists("%s/%f"%(filepath,beta_coef)):
            # If the folder does not exist, create it
            os.makedirs("%s/%f"%(filepath,beta_coef))

        # Convert lists to numpy arrays
        p_values_oracle = np.array(p_values_oracle)
        p_values_median = np.array(p_values_median)
        p_values_LR = np.array(p_values_LR)
        p_values_xgboost = np.array(p_values_xgboost)

        # Save numpy arrays to files
        np.save('%s/%f/p_values_oracle_%d.npy' % (filepath, beta_coef, task_id), p_values_oracle)
        np.save('%s/%f/p_values_median_%d.npy' % (filepath, beta_coef, task_id), p_values_median)
        np.save('%s/%f/p_values_LR_%d.npy' % (filepath, beta_coef,task_id), p_values_LR)
        np.save('%s/%f/p_values_xgboost_%d.npy' % (filepath, beta_coef,task_id), p_values_xgboost)      




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

    for coef in np.arange(0.01,0.2,0.01):
        beta_coef = coef
        run(1000, Unobserved = 0, Single = 1 , filepath = "Result/HPC_power_1000" + "_single")
        run(1000, Unobserved = 1, Single = 1, filepath = "Result/HPC_power_unobserved_1000" + "_single")
        run(2000, Unobserved = 1, Single = 1, filepath = "Result/HPC_power_unobserved_2000" + "_single")
        run(2000, Unobserved = 0, Single = 1 , filepath = "Result/HPC_power_2000" + "_single")
        run(2000, Unobserved = 1, Single = False, filepath = "Result/HPC_power_unobserved_2000" + "_multi")
        run(2000, Unobserved = 0, Single = False , filepath = "Result/HPC_power_2000" + "_multi")
        run(1000, Unobserved = 1, Single = False , filepath = "Result/HPC_power_unobserved_1000" + "_multi")
        run(1000, Unobserved = 0, Single = False, filepath = "Result/HPC_power_1000" + "_multi")  

        


