
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
from sklearn.base import clone
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
import xgboost as xgb
import time


class RetrainTest:
    #load data
    def __init__(self,N,covariance_adjustment = 0):
        self.N = N
        self.covariance_adjustment = covariance_adjustment # 0 = Null, 1 = LR, 2 = XG

    def holm_bonferroni(self,p_values, alpha = 0.05):
        # Perform the Holm-Bonferroni correction
        reject, corrected_p_values, _, _ = multipletests(p_values, alpha=alpha, method='holm')

        # Check if any null hypothesis can be rejected
        any_rejected = any(reject)

        return any_rejected

    def getY(self, G, df_Z, df_noZ, indexY ,lenY):
        if G:
            df_imputed = G.fit_transform(df_Z)
            
        else:
            df_imputed = df_Z.to_numpy()

        #df_noZ_imputed = df_noZ.to_numpy()
        if self.covariance_adjustment == 0:
            return df_imputed[:,indexY:indexY+lenY]
        if self.covariance_adjustment == 1:
            G2 = IterativeImputer(estimator = linear_model.BayesianRidge(),max_iter=3) #IterativeImputer(estimator = xgb.XGBRegressor(),max_iter=3)#
            df_noZ_imputed = G2.fit_transform(df_noZ)
            return df_imputed[:,indexY:indexY+lenY] - df_noZ_imputed[:,indexY-1:indexY+lenY-1]
        if self.covariance_adjustment == 2:
            G2 = IterativeImputer(estimator = xgb.XGBRegressor(),max_iter=3)
            df_noZ_imputed = G2.fit_transform(df_noZ)
            return df_imputed[:,indexY:indexY+lenY] - df_noZ_imputed[:,indexY-1:indexY+lenY-1]

    def get_corr(self, G, df, Y, indexY, lenY):
        # Get the imputed data Y and indicator Z
        if G is None:
            df_imputed = df.to_numpy()
        else:
            df_imputed = G.transform(df)
        y = df_imputed[:, indexY:indexY+lenY]

        # Initialize the lists to store imputed and truth values for missing positions
        y_imputed = []
        y_truth = []

        # Iterate over the rows and columns to find missing values
        for i in range(df.shape[0]):
            for j in range(lenY):
                # Check if the value in the last three columns of df is missing (you can replace 'your_missing_value' with the appropriate value or condition)
                if np.isnan(df.iloc[i, -lenY + j]):
                    y_imputed.append(y[i, j])
                    y_truth.append(Y[i, j])
        if y_imputed == y_truth:
            return 1.0

        # Calculate the correlation between the imputed data and the observed data
        corr = 0.0
        if len(y_imputed) > 0 and len(y_truth) > 0:
            val = np.corrcoef(y_imputed, y_truth)[0, 1]
            if np.isnan(val) == False:
                corr = val

        return corr

    def CombinedT(self,z,y):

        #the Wilcoxon rank sum test
        n = len(z)
        t = 0

        #O(N*Log(N)) version
        my_list = []
        for i in range(n):
            my_list.append((z[i],y[i]))
        sorted_list = sorted(my_list, key=lambda x: x[1])

        #Calculate
        for i in range(n):
            t += sorted_list[i][0] * (i + 1)
        return t

    def getCombinedT(self, y, z, lenY):
        t = []
        for i in range(lenY):
            #the Wilcoxon rank sum test
            t.append(self.T(z.reshape(-1,),y[:,i].reshape(-1,)))
        return t

    def T(self,z,y):

        #the Wilcoxon rank sum test
        n = len(z)
        t = 0
        #O(N^2) version
        """
        for n in range(N):
            rank = sum(1 for n_prime in range(N) if Y[n] >= Y[n_prime])
            T += Z[n] * rank
        """

        #O(N*Log(N)) version
        my_list = []
        for i in range(n):
            my_list.append((z[i],y[i]))
        sorted_list = sorted(my_list, key=lambda x: x[1])

        #Calculate
        for i in range(n):
            t += sorted_list[i][0] * (i + 1)
        
        return t

    def split(self, y, z, M):
        #print(y,z,M)
        missing_indices = M[:].astype(bool)
        non_missing_indices = ~missing_indices

        y_missing = y[missing_indices].reshape(-1,)
        y_non_missing = y[non_missing_indices].reshape(-1,)

        z_missing = z[missing_indices].reshape(-1,)
        z_non_missing = z[non_missing_indices].reshape(-1,)

        return y_missing, y_non_missing, z_missing, z_non_missing

    def getT(self, y, z, lenY, M, verbose = False):

        t = []
        for i in range(lenY):
            # Split the data into missing and non-missing parts using the split function
            y_missing, y_non_missing, z_missing, z_non_missing = self.split(y[:,i], z, M[:,i])
            
            # Calculate T for missing and non-missing parts
            t_missing = self.T(z_missing, y_missing.reshape(-1,))
            t_non_missing = self.T(z_non_missing, y_non_missing.reshape(-1,))

            # Sum the T values for both parts
            t_combined = t_missing + t_non_missing
            if verbose:
                print("t_non_missing:",t_non_missing)
                print("t_missing:",t_missing)
            t.append(t_combined)

        return t

    def retrain_test(self, Z, X, M, Y, G, strata_size, L=10000, verbose = False, shuffle = False):

        if shuffle:
            df = pd.DataFrame({
                'X1': X[:, 0],
                'X2': X[:, 1],
                'X3': X[:, 2],
                'X4': X[:, 3],
                'X5': X[:, 4],
                'Y': Y.flatten(),
                'M': M.flatten(),
                'Z': Z.flatten(),
            })

            # Shuffle the DataFrame
            df = df.sample(frac=1).reset_index(drop=True)

            # Split the shuffled DataFrame back into the individual variables
            Z = df['Z'].values.reshape(-1, 1)
            X = df[['X1', 'X2', 'X3', 'X4', 'X5']].values  # this will create a 2D array for X
            M = df['M'].values.reshape(-1, 1)
            Y = df['Y'].values.reshape(-1, 1)

        if G == None:
            return self.retrain_test_oracle(Z, X, M, Y, G,strata_size, L, verbose)   
        else:
            return self.retrain_test_imputed(Z, X, M, Y, G,strata_size, L, verbose)
        
    def retrain_test_oracle(self, Z, X, M, Y, G, strata_size, L=10000, verbose = False):
        start_time = time.time()

        df_Z = pd.DataFrame(np.concatenate((Z, X, Y), axis=1))

        Y_copy = np.ma.masked_array(Y, mask=M)
        Y_copy = Y_copy.filled(np.nan)

        df_noZ = pd.DataFrame(np.concatenate((X, Y_copy), axis=1))

        # lenY is the number of how many columns are Y
        lenY = Y.shape[1]

        # indexY is the index of the first column of Y
        indexY = Z.shape[1] + X.shape[1]

        # N is the number of rows of the data frame
        N = df_Z.shape[0]

        # re-impute the missing values and calculate the observed test statistics in part 2
        bias = self.getY(G, df_Z, df_noZ, indexY, lenY)
        t_obs = self.getT(bias, Z, lenY, M, verbose = verbose)

        #print train end
        if verbose:
            print("t_obs:"+str(t_obs))
            print("Permutation Start")

        # simulate data and calculate test statistics
        t_sim = np.zeros((L,Y.shape[1]))

        for l in range(L):
            
            # simulate treatment indicators
            Z_sim = []
            half_strata_size = strata_size // 2  # Ensure strata_size is even

            for i in range(int(N/strata_size)):
                strata = np.array([0.0]*half_strata_size + [1.0]*half_strata_size)
                np.random.shuffle(strata)
                Z_sim.append(strata)
            Z_sim = np.concatenate(Z_sim).reshape(-1, 1) 

            df_Z = pd.DataFrame(np.concatenate((Z_sim, X, Y), axis=1))
            bias = self.getY(G, df_Z, df_noZ, indexY, lenY)

            # get the test statistics 
            t_sim[l] = self.getT(bias, Z_sim, lenY, M, verbose=False)

        if verbose:
            print("t_sims_mean:"+str(np.mean(t_sim)))
            print("\n")

        # perform Holm-Bonferroni correction
        p_values = []
        for i in range(lenY):
            p_values.append(np.mean(t_sim[:,i] >= t_obs[i], axis=0))
        reject = self.holm_bonferroni(p_values)

        end_time = time.time()
        test_time = end_time - start_time

        return p_values, reject, test_time

    def retrain_test_imputed(self, Z, X, M, Y, G,  strata_size, L=10000, verbose = False):
        """
        A retrain framework for testing H_0.

        Args:
        Z: 2D array of observed treatment indicators
        X: 2D array of observed covariates
        M: 2D array of observed missing indicators
        Y: 2D array of observed values for K outcomes
        G: a function that takes (Z, X, M, Y_k) as input and returns the imputed value for outcome k
        L: number of Monte Carlo simulations (default is 10000)
        verbose: a boolean indicating whether to print training start and end (default is False)

        Returns:
        p_values: a 1D array of p-values for lenY outcomes
        reject: a boolean indicating whether the null hypothesis is rejected for each outcome
        corr: a 1D array of correlations between the imputed and observed values for lenY outcomes

        """
        start_time = time.time()
        # mask Y
        Y = np.ma.masked_array(Y, mask=M)
        Y = Y.filled(np.nan)

        df_Z = pd.DataFrame(np.concatenate((Z, X, Y), axis=1))


        df_noZ = pd.DataFrame(np.concatenate((X, Y), axis=1))
        G_model = clone(G)

        # lenY is the number of how many columns are Y
        lenY = Y.shape[1]

        # indexY is the index of the first column of Y
        indexY = Z.shape[1] + X.shape[1]

        # N is the number of rows of the data frame
        N = df_Z.shape[0]

        # re-impute the missing values and calculate the observed test statistics in part 2
        bias = self.getY(G, df_Z, df_noZ, indexY, lenY)
        t_obs = self.getT(bias, Z, lenY, M, verbose = verbose)

        # get the correlation of G1 and G2
        corr_G = self.get_corr(G, df_Z, Y, indexY, lenY)

        #print train end
        if verbose:
            print("t_obs:"+str(t_obs))
            print("corr_G:"+str(corr_G))
            print("Permutation Start")

        # simulate data and calculate test statistics
        t_sim = np.zeros((L,Y.shape[1]))

        for l in range(L):
            
            # simulate treatment indicators
            #Z_sim = np.random.binomial(1, 0.5, N).reshape(-1, 1)

            Z_sim = []
            half_strata_size = strata_size // 2  # Ensure strata_size is even

            for i in range(int(N/strata_size)):
                strata = np.array([0.0]*half_strata_size + [1.0]*half_strata_size)
                np.random.shuffle(strata)
                Z_sim.append(strata)
            Z_sim = np.concatenate(Z_sim).reshape(-1, 1) 
            
            G_clone = clone(G_model)
            df_Z = pd.DataFrame(np.concatenate((Z_sim, X, Y), axis=1))
            bias = self.getY(G_clone, df_Z, df_noZ, indexY, lenY)

            # get the test statistics 
            t_sim[l] = self.getT(bias, Z_sim, lenY, M, verbose=False)


        if verbose:
            print("t_sims_mean:"+str(np.mean(t_sim)))
            print("\n")
            print("time:"+str(time.time() - start_time))

        # perform Holm-Bonferroni correction
        p_values = []
        for i in range(lenY):
            p_values.append(np.mean(t_sim[:,i] >= t_obs[i], axis=0))
        reject = self.holm_bonferroni(p_values,alpha = 0.05)

        end_time = time.time()
        test_time = end_time - start_time

        return p_values, reject, test_time