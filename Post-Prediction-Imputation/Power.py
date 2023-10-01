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
L = 1

S_size = 10

def run(Nsize, Unobserved, Single, filepath, adjust, strata_size, Missing_lambda = None,small_size = True, verbose=1):

    # If the folder does not exist, create it
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # Create an instance of the OneShot class
    Framework = Retrain.RetrainTest(N = Nsize, covariance_adjustment=adjust)

    # Simulate data
    DataGen = Generator.DataGenerator(N = Nsize, strata_size=S_size,beta_11 = beta_coef, beta_12 = beta_coef, beta_21 = beta_coef, beta_22 = beta_coef, beta_23 = beta_coef, beta_31 = beta_coef, beta_32 = beta_coef, MaskRate=0.5,Unobserved=Unobserved, Single=Single, verbose=verbose,Missing_lambda = Missing_lambda)

    X, Z, U, Y, M, S = DataGen.GenerateData()
    correlation_matrix = np.corrcoef(Y, rowvar=False)
    print("Correlation matrix of Y:")
    print(correlation_matrix)
    #exit()
    #sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    #plt.show()

    print(X.shape, Z.shape, U.shape, Y.shape, M.shape, S.shape)

    #Oracale imputer
    print("Oracle")
    p_values, reject, test_time = Framework.retrain_test(Z, X, M, Y,strata_size = strata_size, L=L, G = None,verbose=verbose)
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
        print("XGBoost")
        XGBoost = IterativeImputer(estimator=xgb.XGBRegressor(n_jobs=1), max_iter=max_iter)
        p_values, reject, test_time = Framework.retrain_test(Z, X, M, Y, strata_size = strata_size,L=L, G=XGBoost, verbose=verbose)
        values_xgboost = [*p_values, reject, test_time]

    #LightGBM
    if small_size == False:
        print("LightGBM")
        LightGBM = IterativeImputer(estimator=lgb.LGBMRegressor(n_jobs=1,verbosity=-1), max_iter=max_iter)
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

    # Lambda values dictionary
    lambda_values = {
        50: {
            0.0: [5.46301136050662, 1.7687104800990539, 3.6986401066938748],
            0.12: [5.507071138438006, 1.8832179319883895, 3.8250507348009557],
            0.24: [5.629938938721568, 1.9080170719416063, 3.870429428753654],
            0.36: [5.709076777442875, 1.9590050193610664, 4.018691917409632],
            0.48: [5.831068183224691, 1.9638039860442473, 4.032046646076915],
            0.6: [5.890152740793354, 2.0340630188325295, 4.188578787477003]
        },
        1000: {
            0.0: [5.445126353777186, 1.7944628138115826, 3.6890049854144222],
            0.03: [5.448889434968681, 1.799820386146107, 3.69899976186121],
            0.06: [5.481108645836731, 1.808888277601773, 3.7215141167626897],
            0.09: [5.518540969761793, 1.8313022068804186, 3.7592034824941227],
            0.12: [5.509295189307611, 1.824491093343858, 3.7653155995566836],
            0.15: [5.5323113856789075, 1.829439262086321, 3.7932522695382818]
        }
    }
    # 1000 size coef loop
    for coef in np.arange(0.09, 0.18, 0.03): 
        beta_coef = coef
        run(1000, Unobserved=1, Single=0, filepath="Result/HPC_power_1000_unobserved" + "_multi", adjust=0, strata_size=S_size, Missing_lambda=lambda_values[1000].get(coef, None), small_size=False)
        exit()
    # 50 size coef loop
    for coef in np.arange(0.0, 0.72, 0.12): 
        beta_coef = coef
        run(50, Unobserved=1, Single=0, filepath="Result/HPC_power_50_unobserved" + "_multi", adjust=0, strata_size=S_size, Missing_lambda=lambda_values[50].get(coef, None), small_size=True)

