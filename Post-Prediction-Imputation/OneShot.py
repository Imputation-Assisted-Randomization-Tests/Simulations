
import pandas as pd
import numpy as np
import multiprocessing
from statsmodels.stats.multitest import multipletests

class OneShotTest:
    #load data
    def __init__(self,N,Single):
        self.N = N
        self.Single = Single

    #split based on strata
    def split_df(self,df):

        # Sort the groups by the number of rows in each group
        #sorted_df = df.sort_values(by = index_S, ascending=True)
        
        # Split the sorted groups into two equal-sized sets of 100 strata each
        df_set1 = df.iloc[:int(self.N/2), :]
        df_set2 = df.iloc[int(self.N/2):self.N, :]

        #set the index of the two sets from zero to 1
        df_set1.index = range(len(df_set1))
        df_set2.index = range(len(df_set2))
        
        # Return the two sets of strata
        return df_set1, df_set2

    def holm_bonferroni(self,p_values):
        # Perform the Holm-Bonferroni correction
        reject, corrected_p_values, _, _ = multipletests(p_values, alpha=0.05, method='holm')

        # Check if any null hypothesis can be rejected
        any_rejected = any(reject)

        return any_rejected

    def get_corr(self, G, df, indexY, Y,lenY):
        # Get the imputed data Y and indicator Z
        if G is None:
            df_imputed = df.to_numpy()
        else:
            df.to_csv('sample/before_corr_df.csv', index=False)
            df_imputed = G.transform(df)
        
        y = df_imputed[:, indexY:indexY+lenY]
        

        # Initialize the lists to store imputed and truth values for missing positions
        y_imputed = [[] for _ in range(Y.shape[1])]
        y_truth = [[] for _ in range(Y.shape[1])]

        # Iterate over the rows and columns to find missing values
        for i in range(df.shape[0]):
            for j in range(lenY):
                # Check if the value in the last three columns of df is missing (you can replace 'your_missing_value' with the appropriate value or condition)
                if np.isnan(df.iloc[i, -lenY + j]):
                    y_imputed[j].append(y[i, j])
                    y_truth[j].append(Y[i, j])
        if y_imputed == y_truth:
            return [1,1,1]
        

        df = pd.DataFrame(y_imputed)
        df.to_csv('sample/y_imputed.csv')
        df = pd.DataFrame(y_truth)
        df.to_csv('sample/y_truth.csv')
        exit()
        # Convert the lists to NumPy arrays
        y_imputed = [np.array(arr) for arr in y_imputed]
        y_truth = [np.array(arr) for arr in y_truth]

        # Calculate the correlation between the imputed data and the observed data
        corr = []
        for i in range(lenY):
            if len(y_imputed[i]) > 0 and len(y_truth[i]) > 0:
                val = np.corrcoef(y_imputed[i], y_truth[i])[0, 1]
                if np.isnan(val):
                    corr.append(0)
                else:
                    corr.append(val)
            else:
                corr.append(None)
        return corr

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

    def getT(self, G, df, indexY, lenY, M):
        if G is None:
            df_imputed = df.to_numpy()
            y = df_imputed[:, indexY:indexY + lenY]
            z = df_imputed[:, 0]
        else:
            df.to_csv('sample/before_get_T_df.csv', index=False)
            df_imputed = G.transform(df)
            y = df_imputed[:, indexY:indexY + lenY]
            z = df_imputed[:, 0]


        pd.pandas.DataFrame(df_imputed).to_csv('sample/df_imputed.csv')

        t = []
        for i in range(lenY):
            # Split the data into missing and non-missing parts using the split function
            y_missing, y_non_missing, z_missing, z_non_missing = self.split(y[:,i], z, M[:,i])
            
            # Calculate T for missing and non-missing parts
            t_missing = self.T(z_missing, y_missing.reshape(-1,))
            t_non_missing = self.T(z_non_missing, y_non_missing.reshape(-1,))

            # Sum the T values for both parts
            t_combined = t_missing + t_non_missing
            t.append(t_combined)

        return t


    def one_shot_test(self, Z, X, M, Y, G1, G2,  L=10000, verbose = False):
        """
        A one-shot framework for testing H_0.

        Args:
        Z: 2D array of observed treatment indicators
        X: 2D array of observed covariates
        M: 2D array of observed missing indicators
        Y: 2D array of observed values for K outcomes
        G1: a function that takes (Z, X, M, Y_k) as input and returns the imputed value for outcome k
        G2: a function that takes (Z, X, M, Y_k) as input and returns the imputed value for outcome k
        L: number of Monte Carlo simulations (default is 10000)

        Returns:
        p1: 1D array of exact p-values for testing Fisher's sharp null in part 1
        p2: 1D array of exact p-values for testing Fisher's sharp null in part 2
        """

        #print train start
        if verbose:
            print("Training start")

        # create data a whole data frame
        Y_masked = np.ma.masked_array(Y, mask=M)
        Y_masked = Y_masked.filled(np.nan)
    
        df_oracle = pd.DataFrame(np.concatenate((Z, X, Y), axis=1))
        df = pd.DataFrame(np.concatenate((Z, X, Y_masked), axis=1))
        
        print(df)
        # split the data into two parts
        df1, df2 = self.split_df(df)
        df1_oracle, df2_oracle = self.split_df(df_oracle)
        
        import os
        if os.path.exists("sample") == False:
            os.mkdir("sample")

        # re-impute the missing values and calculate the observed test statistics in part 2
        df2.to_csv("sample/df2.csv")
        df2_oracle.to_csv("sample/df2_oracle.csv")
        t2_obs = np.zeros(Y_masked.shape[1])
        df1.to_csv('sample/df1.csv', index=False)
        G1.fit(df1)
        t2_obs = self.getT(G1, df2, Z.shape[1] + X.shape[1], lenY = Y.shape[1], M = M[df1.shape[0]:df.shape[0],:])

        # get the correlation of G1 and G2
        corr_G1 = self.get_corr(G1, df2, Z.shape[1] + X.shape[1],Y[df1.shape[0]:,:], lenY=Y.shape[1])


        return p11, p12, p21, p22, p31, p32, corr_G1, corr_G2, reject