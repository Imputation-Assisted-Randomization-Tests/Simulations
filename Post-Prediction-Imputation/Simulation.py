import numpy as np
from mv_laplace import MvLaplaceSampler
import pandas as pd
from scipy.stats import logistic


class DataGenerator:
  def __init__(self,*, N = 1000, strata_size = 10, beta_11 = 1, beta_12 = 1, beta_21 = 1, beta_22 = 1, beta_23 = 1, beta_31 = 1,beta_32 = 1, MaskRate = 0.3, Unobserved = True, Single = True,  verbose = False, bias = False, Missing_lambda= None):
    self.N = N
    self.beta_11 = beta_11
    self.beta_12 = beta_12
    self.beta_21 = beta_21
    self.strata_size = strata_size
    self.totalStrataNumber = int(N/strata_size)
    self.beta_22 = beta_22
    self.beta_23 = beta_23
    self.beta_31 = beta_31
    self.beta_32 = beta_32
    self.MaskRate = MaskRate
    self.Unobserved = Unobserved
    self.Single = Single
    self.verbose = verbose
    self.bias = bias
    self.Missing_lambda = Missing_lambda
    
  
  def GenerateX(self):

      # generate Xn1 and Xn2
      mean = [1/2, -1/3]
      cov = [[1, 1/2], [1/2, 1]]
      X1_2 = np.random.multivariate_normal(mean, cov, self.N)

      # generate Xn3 and Xn4
      loc = [0, 1/np.sqrt(3)]
      cov = [[1,1/np.sqrt(2)], [1/np.sqrt(2),1]]

      sampler = MvLaplaceSampler(loc, cov)
      X3_4 = sampler.sample(self.N)

      # generate Xn5
      p = 1/3
      X5 = np.random.binomial(1, p, self.N)

      # combine all generated variables into a single matrix
      X = np.hstack((X1_2, X3_4, X5.reshape(-1,1)))
      
      return X

  def GenerateU(self):
      # generate U
      mean = 0
      std = 0.5
      U = np.random.normal(mean, std, self.N)
      U = U.reshape(-1, 1)
      return U

  def GenerateS(self):
    # Add strata index
    S = np.zeros(self.N)
    for i in range(self.totalStrataNumber):
        S[self.strata_size*i:self.strata_size*(i+1)] = i + 1
    S = S.reshape(-1, 1)
    return S

  def GenerateZ(self):
    Z = []
    half_strata_size = self.strata_size // 2  # Ensure strata_size is even

    for i in range(self.totalStrataNumber):
        strata = np.array([0.0]*half_strata_size + [1.0]*half_strata_size)
        np.random.shuffle(strata)
        Z.append(strata)
    Z = np.concatenate(Z).reshape(-1, 1) 

    return Z

  def GenerateIndividualEps(self):
      mean = 0
      std = 0.2
      eps = np.random.normal(mean, std, (self.N, 3))
      return eps

  def GenerateStrataEps(self):
      eps1 = []
      eps2 = []
      eps3 = []

      for i in range(self.totalStrataNumber):
          eps1.append(np.full((self.strata_size), np.random.normal(0, 0.1)))
          eps2.append(np.full((self.strata_size), np.random.normal(0, 0.1)))
          eps3.append(np.full((self.strata_size), np.random.normal(0, 0.1)))

      eps1 = np.concatenate(eps1)
      eps2 = np.concatenate(eps2)
      eps3 = np.concatenate(eps3)
      eps = np.column_stack((eps1, eps2, eps3))
      return eps

  def GenerateY(self, X, U, Z,  StrataEps, IndividualEps):
        
    #def sum1():
    sum1 = np.zeros(self.N)
    for p in range(1,6):
      sum1 += logistic.cdf(X[:,p-1])
    sum1 = (1.0 / np.sqrt(5)) * sum1

    #def sum2():
    sum2 = np.zeros(self.N)
    for p in range(1,6):
      for p_2 in range(1,6):
        sum2 += X[:,p-1] * X[:,p_2-1]
    sum2 = (1.0 / np.sqrt(5 * 5)) * sum2

    #def sum3():
    sum3 = np.zeros(self.N)
    for p in range(1,6):
      sum3 += X[:,p-1]
    sum3 = (1.0 / np.sqrt(5)) * sum3

    #def sum4():
    sum4 = np.zeros(self.N)
    for p in range(1,6):
      for p_2 in range(1,6):
        sum4 += X[:,p-1] * np.sin(1 - X[:,p_2-1])
    sum4 = (1.0 / np.sqrt(5 * 5)) * sum4

    #def sum5():
    sum5 = np.zeros(self.N)
    for p in range(1,6):
      sum5 += np.absolute(X[:,p-1])
    sum5 = (1.0  / np.sqrt(5)) * sum5

    #def sum6(): 
    sum6 = np.zeros(self.N)
    for p in range(1,6):
      sum6 += X[:,p-1]
    sum6 = (1.0  / np.sqrt(5)) * sum6

    sum9 = np.zeros(self.N)
    for p in range(1,6):
      for p_2 in range(1,6):
        sum9 += X[:,p-1] * np.cos(X[:,p_2-1])
    sum9 = (1.0 / np.sqrt(5 * 5)) * sum9
    #def sum8(): 
    sum8 = np.zeros(self.N)
    for p in range(1,6):
      for p_2 in range(1,6):
        for p_3 in range(1,6):
          sum8 += X[:,p-1] * X[:,p_2-1] * np.cos(1 - 4*X[:,p_3-1])
    sum8 = (1.0  / np.sqrt(5 * 5 )) * sum8  

    U = U.reshape(-1,)
    Z = Z.reshape(-1,)
    
    # Calculate Y_n1
    Y_n1 = 1/4 * self.beta_11 * Z + self.beta_12 * Z * sum1   + sum2 + sum3 + np.sin(U) + StrataEps[:,0]  + IndividualEps[:,0]

    # Compute Yn2
    Y_n2 = self.beta_11 * Z + self.beta_22 * Z * (X[:,0])**2  +  sum9 - sum4 + StrataEps[:,1] + IndividualEps[:,1]

    # Compute Yn3
    Y_n3 = self.beta_11 * Z + self.beta_32 * Z * sum5  + sum6 + sum8 + U +  StrataEps[:,2]  + IndividualEps[:,2]
  
    Y = np.concatenate((Y_n1.reshape(-1, 1), Y_n2.reshape(-1, 1),Y_n3.reshape(-1, 1)), axis=1) 
  
    return Y

  def GenerateM(self, X, U, Y, single = True):
      

      U = U.reshape(-1,)
      n = X.shape[0]

      if not single:

        M = np.zeros((n, 3))
        M_lamda = np.zeros((n, 3))

        #for verbose
        if self.verbose:
          M_lamda_Y = np.zeros((n, 3))
          M_lamda_X = np.zeros((n, 3))
          M_lamda_U = np.zeros((n, 3))

        for i in range(n):
            sum1 = 0
            for p in range(1,6):
              for p_2 in range(1,6):
                sum1 += X[i,p-1] * X[i,p_2-1]
            sum1 = (1.0  / np.sqrt(5 * 5)) * sum1
            
            sum2 = 0
            for p in range(1,6):
              for p_2 in range(1,6):
                sum2 += X[i,p-1] * X[i,p_2-1]
            sum2 = (1.0  / np.sqrt(5 * 5)) * sum2

            sum3 = 0
            for p in range(1,6):
                sum3 += p * X[i,p-1] 
            sum3 = (1.0  / np.sqrt(5 * 5)) * sum3

            sum4 = 0
            for p in range(1,6):
              for p_2 in range(1,6):
                for p_3 in range(1,6):
                  sum4 += X[i,p-1] * X[i,p_2-1] * X[i,p_3-1]
            sum4 = (1.0  / np.sqrt(5 * 5 * 5)) * sum4

            M_lamda[i][0] = (1.0  / np.sqrt(5))* logistic.cdf(X[i, :]).sum() + sum1 + 5 * logistic.cdf(Y[i, 0] )

            M_lamda[i][1] = (1.0  / np.sqrt(5))*((X[i, :]**3).sum() + sum2  + 2.5 * logistic.cdf(Y[i, 0]) +2.5 * logistic.cdf(Y[i, 1]))

            M_lamda[i][2] = (sum3 + sum4 + 5/3 * logistic.cdf(Y[i, 0]) +  5/3 * logistic.cdf(Y[i, 1]) + 5/3 * logistic.cdf(Y[i, 2]) )


            if self.verbose:
              M_lamda_Y[i][0] =  5 * logistic.cdf(Y[i, 0])
              M_lamda_X[i][0] = (1.0  / np.sqrt(5))* logistic.cdf(X[i, :]).sum() + sum1
              M_lamda_U[i][0] = np.sin(U[i])**3 

              M_lamda_Y[i][1] =  2.5 * logistic.cdf(Y[i, 0]) + 2.5 * logistic.cdf(Y[i, 1])
              M_lamda_X[i][1] = (1.0  / np.sqrt(5))*((X[i, :]**3).sum() + sum2 )
              M_lamda_U[i][1] = (U[i] )

              M_lamda_Y[i][2] = 5/3 * logistic.cdf(Y[i, 0]) +  5/3 * logistic.cdf(Y[i, 1]) + 5/3 * logistic.cdf(Y[i, 2])
              M_lamda_X[i][2] = (sum3 + sum4)
              M_lamda_U[i][2] = np.sin(U[i]) 

        if self.verbose:
          data = pd.DataFrame(M_lamda_Y, columns=['Y1', 'Y2', 'Y3'])
          data['X1'] = M_lamda_X[:,0]
          data['X2'] = M_lamda_X[:,1]
          data['X3'] = M_lamda_X[:,2]
          data['U1'] = M_lamda_U[:,0]
          data['U2'] = M_lamda_U[:,1]
          data['U3'] = M_lamda_U[:,2]
          print(data.describe())

        # calculate 1 - Maskrate percentile
        if self.Missing_lambda:
          lambda1 = self.Missing_lambda[0]
          lambda2 = self.Missing_lambda[1]
          lambda3 = self.Missing_lambda[2]

        #else:
          #lambda1 = np.percentile(M_lamda[:,0], 100 * (1-self.MaskRate))
          #lambda2 = np.percentile(M_lamda[:,1], 100 * (1-self.MaskRate))
          #lambda3 = np.percentile(M_lamda[:,2], 100 * (1-self.MaskRate))

            
        for i in range(n):
            values = np.zeros(3)
            sum1 = 0
            for p in range(1,6):
              for p_2 in range(1,6):
                sum1 += X[i,p-1] * X[i,p_2-1]
            sum1 = (1.0  / np.sqrt(5 * 5)) * sum1
            
            sum2 = 0
            for p in range(1,6):
              for p_2 in range(1,6):
                sum2 += X[i,p-1] * X[i,p_2-1]
            sum2 = (1.0  / np.sqrt(5 * 5)) * sum2

            sum3 = 0
            for p in range(1,6):
                sum3 += p * X[i,p-1] 
            sum3 = (1.0  / np.sqrt(5 * 5)) * sum3

            sum4 = 0
            for p in range(1,6):
              for p_2 in range(1,6):
                for p_3 in range(1,6):
                  sum4 += X[i,p-1] * X[i,p_2-1] * X[i,p_3-1]
            sum4 = (1.0  / np.sqrt(5 * 5 * 5)) * sum4

            values[0] = (1.0  / np.sqrt(5))* logistic.cdf(X[i, :]).sum() + sum1 + 5 * logistic.cdf(Y[i, 0])

            values[1] = (1.0  / np.sqrt(5))*((X[i, :]**3).sum() + sum2  + 5/2 * logistic.cdf(Y[i, 0]) + 5/2 * logistic.cdf(Y[i, 1]))
            values[2] = (sum3 + sum4 + 5/3 * logistic.cdf(Y[i, 0]) +  5/3 * logistic.cdf(Y[i, 1]) + 5/3 * logistic.cdf(Y[i, 2]) )

            M[i][0] = (values[0] > lambda1)
            M[i][1] =  (values[1] > lambda2)
            M[i][2] =  (values[2] > lambda3)
       
        print(pd.DataFrame(M).describe())
        return M
  
  
  def GenerateData(self):  
    # Generate X
    X = self.GenerateX()

    # Generate U
    U = self.GenerateU()

    # Generate S
    S = self.GenerateS()

    # Generate Z
    Z = self.GenerateZ()

    # generate Interf
    StrataEps = self.GenerateStrataEps()
    IndividualEps = self.GenerateIndividualEps()
    
    # Generate Y
    Y = self.GenerateY(X, U, Z, StrataEps, IndividualEps)

    # Generate M
    M = self.GenerateM(X, U, Y, single = self.Single)

    return X, Z, U, Y , M, S

  def StoreData(self,file):
    # Generate data
    X, Z, U, Y, M, S = self.GenerateData()

    # Combine all generated variables into a single matrix
    data = np.concatenate((X, Z, U, Y, M, S), axis=1) 

    # Store data
    np.savetxt(file, data, delimiter=",")

    # Print message
    print("Data stored in SimulatedData.csv")

  def ReadData(self,file):
    # Read data
    data = np.genfromtxt(file, delimiter=',')
    # Splite into X, Z, U, Y, M, S
    X = data[:, :5]
    Z = data[:, 5]
    U = data[:, 6:8]
    Y = data[:, 8:11]
    M = data[:, 11:14]
    S = data[:, 14]

    return X, Z, U, Y, M, S