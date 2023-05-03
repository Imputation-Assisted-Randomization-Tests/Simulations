
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

        print(sorted_list)
        #Calculate
        for i in range(n):
            t += sorted_list[i][0] * (i + 1)
        
        return np.array(t)

    def getT(self, G, df, indexY, lenY):
        
        if G is None:
            # Get the imputed data Y and indicator Z
            #df_imputed = np.array(df)
            df_imputed = df.to_numpy()
            y = df_imputed[:, indexY:indexY + lenY]
            z = df_imputed[:, 0]
            pd.DataFrame(df_imputed).to_csv('sample/df_imputed_oracle.csv')
            pd.DataFrame(z).to_csv('sample/z_oracle.csv')
            pd.DataFrame(y).to_csv('sample/y_oracle.csv')
        else:
            # Get the imputed data Y and indicator Z
            df_imputed = G.transform(df)
            y = df_imputed[:, indexY:indexY + lenY]
            z = df_imputed[:, 0]
            pd.DataFrame(df_imputed).to_csv('sample/df_imputed_imputation.csv')
            pd.DataFrame(z).to_csv('sample/z_imputation.csv')
            pd.DataFrame((y)).to_csv('sample/y_imputation.csv')
        
        t = []
        for i in range(lenY):
            #the Wilcoxon rank sum test
            t.append(self.T(z.reshape(-1,),y[:,i].reshape(-1,)))
        
        return t

    def worker(self, args):
        # unpack the arguments
        X, Y, Y_masked, G1, G2, t1_obs, t2_obs, L, verbose = args

        # simulate data and calculate test statistics
        t1_sim = np.zeros((L,Y_masked.shape[1]))
        t2_sim = np.zeros((L,Y_masked.shape[1]))

        for l in range(L):

            # simulate treatment indicators in parts 1 and 2
            if G1 is None or G2 is None:
                df_sim = pd.DataFrame(np.concatenate((X, Y), axis=1))
            else:
                df_sim = pd.DataFrame(np.concatenate((X, Y_masked), axis=1))
            
            # split the simulated data into two parts
            df1_sim, df2_sim = self.split_df(df_sim)

            # simulate treatment indicators in parts 1 and 2
            Z_1 = np.random.binomial(1, 0.5, df1_sim.shape[0]).reshape(-1, 1)
            Z_2 = np.random.binomial(1, 0.5, df2_sim.shape[0]).reshape(-1, 1)
            df1_sim = pd.concat([pd.DataFrame(Z_1), df1_sim], axis=1)
            df2_sim = pd.concat([pd.DataFrame(Z_2), df2_sim], axis=1)

            # get the test statistics in part 1
            t1_sim[l] = self.getT(G2, df1_sim, Z_1.shape[1] + X.shape[1], lenY = Y_masked.shape[1])

            # get the test statistics in part 2
            t2_sim[l] = self.getT(G1, df2_sim, Z_2.shape[1] + X.shape[1], lenY = Y_masked.shape[1])

            # Calculate the completeness percentage
            if l % 100 == 0:
                
                completeness = l / L * 100  
                if verbose:
                    print("t1_sim:"+str(t1_sim[l]), "t2_sim:"+str(t2_sim[l]))
                    print(f"Task is {completeness:.2f}% complete.")

        if self.Single:
            # calculate exact p-values for each outcome
            p11 = 1
            p12 = 1
            p21 = 1
            p22 = 1  
            p31 = np.mean(t1_sim[:,0] >= t1_obs[0], axis=0)
            p32 = np.mean(t2_sim[:,0] >= t2_obs[0], axis=0)
        else:
            # calculate exact p-values for each outcome
            p11 = np.mean(t1_sim[:,0] >= t1_obs[0], axis=0)
            p12 = np.mean(t2_sim[:,0] >= t2_obs[0], axis=0)
            p21 = np.mean(t1_sim[:,1] >= t1_obs[1], axis=0)
            p22 = np.mean(t2_sim[:,1] >= t2_obs[1], axis=0)
            p31 = np.mean(t1_sim[:,2] >= t1_obs[2], axis=0)
            p32 = np.mean(t2_sim[:,2] >= t2_obs[2], axis=0)            
        
        return p11, p12, p21, p22, p31, p32

    def one_shot_test_parallel(self, Z, X, M, Y, G1, G2, L=10000, n_jobs=None,verbose = False):
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

        # Calculate the number of worker processes
        if n_jobs is None or n_jobs < 1:
            n_jobs = multiprocessing.cpu_count()
        #print train start
        if verbose:
            print("Training start")

        # create data a whole data frame
        Y_masked = np.ma.masked_array(Y, mask=M)
        Y_masked = Y_masked.filled(np.nan)

        if G1 == None or G2 == None:
            df = pd.DataFrame(np.concatenate((Z, X, Y), axis=1))
        else:
            df = pd.DataFrame(np.concatenate((Z, X, Y_masked), axis=1))   

        # split the data into two parts
        df1, df2 = self.split_df(df)

        # re-impute the missing values and calculate the observed test statistics in part 2
        t2_obs = np.zeros(Y_masked.shape[1])
        if G1 != None:
            G1.fit(df1)
        t2_obs = self.getT(G1, df2, Z.shape[1] + X.shape[1], lenY = Y.shape[1])

        # re-impute the missing values and calculate the observed test statistics in part 1
        t1_obs = np.zeros(Y_masked.shape[1])
        if G2 != None:
            G2.fit(df2)
        t1_obs = self.getT(G2, df1, Z.shape[1] + X.shape[1], lenY = Y.shape[1])

        # get the correlation of G1 and G2
        if G1 != None:
            corr_G1 = self.get_corr(G1, df2, Z.shape[1] + X.shape[1],Y[df1.shape[0]:,:], lenY=Y.shape[1])
        else:
            corr_G1 = [1,1,1]
        if G2 != None:
            corr_G2 = self.get_corr(G2, df1, Z.shape[1] + X.shape[1],Y[0:df1.shape[0],:], lenY=Y.shape[1])
        else:
            corr_G2 = [1,1,1]
        
        #print train end
        if verbose:
            print("t1_obs:"+str(t1_obs), "t2_obs:"+str(t2_obs))
            print("corr_G1:"+str(corr_G1), "corr_G2:"+str(corr_G2))
            print("Training end")
        
        # print the number of cores
        if verbose:
            print(f"Number of jobs: {n_jobs}")

        # simulate data and calculate test statistics in parallel
        args_list = [(X, Y, Y_masked, G1, G2, t1_obs, t2_obs, int(L / n_jobs), verbose)] * n_jobs
        with multiprocessing.Pool(processes=n_jobs) as pool:
            p_list = pool.map(self.worker, args_list)
        
        p11 = np.mean([p[0] for p in p_list], axis=0)
        p12 = np.mean([p[1] for p in p_list], axis=0)
        p21 = np.mean([p[2] for p in p_list], axis=0)
        p22 = np.mean([p[3] for p in p_list], axis=0)
        p31 = np.mean([p[4] for p in p_list], axis=0)
        p32 = np.mean([p[5] for p in p_list], axis=0)
        
        # perform Holm-Bonferroni correction
        if self.Single:
            reject = self.holm_bonferroni([p31, p32])
            
        else:
            reject = self.holm_bonferroni([p11, p12, p21, p22, p31, p32])

        return p11, p12, p21, p22, p31, p32, corr_G1, corr_G2, reject
    
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
    
        if G1 == None or G2 == None:
            df = pd.DataFrame(np.concatenate((Z, X, Y), axis=1))
        else:
            df = pd.DataFrame(np.concatenate((Z, X, Y_masked), axis=1))
        
        pd.DataFrame(Z).to_csv("sample/Z.csv") if verbose else None
        pd.DataFrame(X).to_csv("sample/X.csv") if verbose else None
        pd.DataFrame(Y).to_csv("sample/Y.csv") if verbose else None
        pd.DataFrame(Y_masked).to_csv("sample/Y_masked.csv") if verbose else None
        
        
        
        # split the data into two parts
        df1, df2 = self.split_df(df)
        if G1 == None or G2 == None:
            df1.to_csv("sample/df_oracle.csv") if verbose else None
        else:
            df1.to_csv("sample/df_missing.csv") if verbose else None

        # re-impute the missing values and calculate the observed test statistics in part 2
        t2_obs = np.zeros(Y_masked.shape[1])
        if G1 != None:
            G1.fit(df1)
        t2_obs = self.getT(G1, df2, Z.shape[1] + X.shape[1], lenY = Y.shape[1])

        # re-impute the missing values and calculate the observed test statistics in part 1
        t1_obs = np.zeros(Y_masked.shape[1])
        if G2 != None:
            G2.fit(df2)
        t1_obs = self.getT(G2, df1, Z.shape[1] + X.shape[1], lenY = Y.shape[1])

        # get the correlation of G1 and G2
        if G1 != None:
            corr_G1 = self.get_corr(G1, df2, Z.shape[1] + X.shape[1],Y[df1.shape[0]:,:], lenY=Y.shape[1])
        else:
            corr_G1 = [1,1,1]
        if G2 != None:
            corr_G2 = self.get_corr(G2, df1, Z.shape[1] + X.shape[1],Y[0:df1.shape[0],:], lenY=Y.shape[1])
        else:
            corr_G2 = [1,1,1]

        #print train end
        if verbose:
            print("t1_obs:"+str(t1_obs), "t2_obs:"+str(t2_obs))
            print("corr_G1:"+str(corr_G1), "corr_G2:"+str(corr_G2))
            print("Training end")

        # simulate data and calculate test statistics
        t1_sim = np.zeros((L,Y.shape[1]))
        t2_sim = np.zeros((L,Y.shape[1]))

        for l in range(L):

            # simulate treatment indicators in parts 1 and 2
            if G1 == None or G2 == None:
                df_sim = pd.DataFrame(np.concatenate((X, Y), axis=1))
            else:
                df_sim = pd.DataFrame(np.concatenate((X, Y_masked), axis=1))
            
            # split the simulated data into two parts
            df1_sim, df2_sim = self.split_df(df_sim)

            # simulate treatment indicators in parts 1 and 2
            Z_1 = np.random.binomial(1, 0.5, df1_sim.shape[0]).reshape(-1, 1)
            Z_2 = np.random.binomial(1, 0.5, df2_sim.shape[0]).reshape(-1, 1)
            df1_sim = pd.concat([pd.DataFrame(Z_1), df1_sim], axis=1)
            df2_sim = pd.concat([pd.DataFrame(Z_2), df2_sim], axis=1)
            

            # get the test statistics in part 2
            t2_sim[l] = self.getT(G1, df2_sim,  Z.shape[1] + X.shape[1], lenY = Y.shape[1])

            # get the test statistics in part 1
            t1_sim[l] = self.getT(G2, df1_sim, Z.shape[1] + X.shape[1], lenY = Y.shape[1])
            
            # Calculate the completeness percentage
            if l % 100 == 0:
                completeness = l / L * 100  
                
                if verbose:
                    print(f"Task is {completeness:.2f}% complete.")
                    print("t1_sim:"+str(t1_sim[l]), "t2_sim:"+str(t2_sim[l]))

        # perform Holm-Bonferroni correction
        if self.Single:
            p11 = 1
            p12 = 1
            p21 = 1
            p22 = 1  
            p31 = np.mean(t1_sim[:,0] >= t1_obs[0], axis=0)
            p32 = np.mean(t2_sim[:,0] >= t2_obs[0], axis=0)
            reject = self.holm_bonferroni([p31, p32])
        else:
            p11 = np.mean(t1_sim[:,0] >= t1_obs[0], axis=0)
            p12 = np.mean(t2_sim[:,0] >= t2_obs[0], axis=0)
            p21 = np.mean(t1_sim[:,1] >= t1_obs[1], axis=0)
            p22 = np.mean(t2_sim[:,1] >= t2_obs[1], axis=0)
            p31 = np.mean(t1_sim[:,2] >= t1_obs[2], axis=0)
            p32 = np.mean(t2_sim[:,2] >= t2_obs[2], axis=0)
            reject = self.holm_bonferroni([p11, p12, p21, p22, p31, p32])

        return p11, p12, p21, p22, p31, p32, corr_G1, corr_G2, reject
