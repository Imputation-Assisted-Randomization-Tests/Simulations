import numpy as np
from mv_laplace import MvLaplaceSampler
import pandas as pd
from scipy.stats import logistic

class DataGenerator:
  def __init__(self,*, N = 1000, N_T = 500, N_S = 50, beta_11 = 1, beta_12 = 1, beta_21 = 1, beta_22 = 1, beta_23 = 1, beta_31 = 1,beta_32 = 1, MaskRate = 0.3, Unobserved = True, Single = True, verbose = False):
    self.N = N
    self.N_T = N_T
    self.N_S = N_S
    self.beta_11 = beta_11
    self.beta_12 = beta_12
    self.beta_21 = beta_21
    self.beta_22 = beta_22
    self.beta_23 = beta_23
    self.beta_31 = beta_31
    self.beta_32 = beta_32
    self.MaskRate = MaskRate
    self.Unobserved = Unobserved
    self.Single = Single
    self.verbose = verbose

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
      std = 1
      U = np.random.normal(mean, std, self.N)
      U = U.reshape(-1, 1)

      return U

  def GenerateS(self):
    # Add strata index
    groupSize = int(self.N / self.N_S)
    S = np.zeros(self.N)
    for i in range(self.N_S):
        S[groupSize*i:groupSize*(i+1)] = i + 1
    S = S.reshape(-1, 1)
    return S

  def GenerateZ(self):
    Z = []
    groupSize = int(self.N / self.N_S)

    for i in range(self.N_S):
        Z.append(np.random.binomial(1, 0.5, groupSize))

    Z = np.concatenate(Z).reshape(-1, 1)
    return Z
  
  def GenerateY(self, X, U, Z):
        
    #def sum1():
    sum1 = np.zeros(self.N)
    for p in range(1,6):
      sum1 += pow(X[:,p-1],3)
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
        sum4 += X[:,p-1] * logistic.cdf(1 - X[:,p_2-1])
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

    #def sum7(): 
    sum7 = np.zeros(self.N)
    for p in range(1,6):
      for p_2 in range(1,6):
        for p_3 in range(1,6):
          sum7 += X[:,p-1] * X[:,p_2-1] * logistic.cdf(X[:,p_3-1])
    sum7 = (1.0  / np.sqrt(5 * 5 * 5)) * sum7

    U = U.reshape(-1,)
    Z = Z.reshape(-1,)

    if self.verbose:
      Y_n1_Z = (self.beta_11 * Z + self.beta_12 * Z * sum1)
      Y_n1_X = sum2
      Y_n1_U = np.sin(U)

      Y_n2_Z = (self.beta_21 * Z + self.beta_22 * Z * X[:,0] + self.beta_23 * Z * U)
      Y_n2_X = sum3 + sum4
      Y_n2_U = self.beta_23 * Z * U

      Y_n3_Z = (self.beta_31 * Z + self.beta_32 * Z * sum5)
      Y_n3_X = sum6 + sum7
      Y_n3_U = U

      data = pd.DataFrame({'Y_n1_Z': Y_n1_Z, 'Y_n1_X': Y_n1_X, 'Y_n1_U': Y_n1_U, 'Y_n2_Z': Y_n2_Z, 'Y_n2_X': Y_n2_X, 'Y_n2_U': Y_n2_U, 'Y_n3_Z': Y_n3_Z, 'Y_n3_X': Y_n3_X, 'Y_n3_U': Y_n3_U})
      print(data.describe())
    
    if self.Unobserved:
      # Calculate Y_n1
      Y_n1 = (self.beta_11 * Z + self.beta_12 * Z * sum1   + sum2 + np.sin(U) )

      # Compute Yn2
      Y_n2 = (self.beta_21 * Z + self.beta_22 * Z * X[:,0] + self.beta_23 * Z * U + sum3 + sum4) 

      # Compute Yn3
      Y_n3 = (self.beta_31 * Z + self.beta_32 * Z * sum5  + sum6 + sum7 + U)

    else:
      # Calculate Y_n1
      Y_n1 = (self.beta_11 * Z + self.beta_12 * Z * sum1  + sum2) 

      # Compute Yn2
      Y_n2 = (self.beta_21 * Z + self.beta_22 * Z * X[:,0] + sum3 + sum4) 

      # Compute Yn3
      Y_n3 = (self.beta_31 * Z + self.beta_32 * Z * sum5 + sum6 + sum7) 
    
    if self.Single:
      Y = Y_n3.reshape(-1, 1)
    else:
      Y = np.concatenate((Y_n1.reshape(-1, 1), Y_n2.reshape(-1, 1),Y_n3.reshape(-1, 1)), axis=1) 

    return Y

  def GenerateM(self, X, U, Y, single = True):
      
      U = U.reshape(-1,)
      n = X.shape[0]

      if single:
        M = np.zeros((n, 1))
        M_lamda = np.zeros((n, 1))
        for i in range(n):
          sum3 = 0
          for p in range(1,6):
              sum3 += p * X[i,p-1] 
          sum3 = (1.0  / np.sqrt(5)) * sum3

          sum4 = 0
          for p in range(1,6):
            for p_2 in range(1,6):
              for p_3 in range(1,6):
                sum4 += X[i,p-1] * X[i,p_2-1] * X[i,p_3-1]
          sum4 = (1.0  / np.sqrt(5 * 5 * 5)) * sum4
          
          M_lamda[i][0] = (sum3 + sum4 + np.sin(U[i])  + Y[i, 0] + logistic.cdf(Y[i, 0]))
        
        lambda1 = np.percentile(M_lamda, 100 * (1-self.MaskRate))

        for i in range(n):
          sum3 = 0
          for p in range(1,6):
              sum3 += p * X[i,p-1] 
          sum3 = (1.0  / np.sqrt(5)) * sum3

          sum4 = 0
          for p in range(1,6):
            for p_2 in range(1,6):
              for p_3 in range(1,6):
                sum4 += X[i,p-1] * X[i,p_2-1] * X[i,p_3-1]
          sum4 = (1.0  / np.sqrt(5 * 5 * 5)) * sum4

          if (sum3 + sum4 + np.sin(U[i]) + Y[i, 0] + logistic.cdf(Y[i, 0])) > lambda1:
            M[i][0] = 1

        return M

      else:
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
            sum3 = (1.0  / np.sqrt(5)) * sum3

            sum4 = 0
            for p in range(1,6):
              for p_2 in range(1,6):
                for p_3 in range(1,6):
                  sum4 += X[i,p-1] * X[i,p_2-1] * X[i,p_3-1]
            sum4 = (1.0  / np.sqrt(5 * 5 * 5)) * sum4

            M_lamda[i][0] = (1.0  / np.sqrt(5))* logistic.cdf(X[i, :]).sum() + sum1 + -1 * logistic.cdf(Y[i, 0])

            M_lamda[i][1] = (1.0  / np.sqrt(5))*((X[i, :]**3).sum() + sum2 + -1 * Y[i, 0] + -1 * Y[i, 1])

            M_lamda[i][2] = (sum3 + sum4 + -1 * Y[i, 0] + -1 * logistic.cdf(Y[i, 1]) + -1 * np.absolute(Y[i, 2]))

            if self.verbose:
              M_lamda_Y[i][0] = logistic.cdf(Y[i, 0])
              M_lamda_X[i][0] = (1.0  / np.sqrt(5))* logistic.cdf(X[i, :]).sum() + sum1
              M_lamda_U[i][0] = np.sin(U[i])**3

              M_lamda_Y[i][1] =  Y[i, 0] + Y[i, 1]
              M_lamda_X[i][1] = (1.0  / np.sqrt(5))*((X[i, :]**3).sum() + sum2 )
              M_lamda_U[i][1] = (U[i])

              M_lamda_Y[i][2] = ( Y[i, 0] + logistic.cdf(Y[i, 1]) + np.absolute(Y[i, 2]))
              M_lamda_X[i][2] = (sum3 + sum4)
              M_lamda_U[i][2] = ( np.sin(U[i]))

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
        lambda1 = np.percentile(M_lamda[:,0], 100 * (1-self.MaskRate))
        lambda2 = np.percentile(M_lamda[:,1], 100 * (1-self.MaskRate))
        lambda3 = np.percentile(M_lamda[:,2], 100 * (1-self.MaskRate))
            
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
            sum3 = (1.0  / np.sqrt(5)) * sum3

            sum4 = 0
            for p in range(1,6):
              for p_2 in range(1,6):
                for p_3 in range(1,6):
                  sum4 += X[i,p-1] * X[i,p_2-1] * X[i,p_3-1]
            sum4 = (1.0  / np.sqrt(5 * 5 * 5)) * sum4
            values[0] = (1.0  / np.sqrt(5))* logistic.cdf(X[i, :]).sum() + sum1  + -1 * logistic.cdf(Y[i, 0])
            values[1] = (1.0  / np.sqrt(5))*((X[i, :]**3).sum() + sum2 + -1 * Y[i, 0] + -1 * Y[i, 1])
            values[2] = (sum3 + sum4 + -1 * Y[i, 0] + -1 * logistic.cdf(Y[i, 1]) + -1 * np.absolute(Y[i, 2]))

            M[i][0] = (values[0] > lambda1)
            M[i][1] =  (values[1] > lambda2)
            M[i][2] =  (values[2] > lambda3)

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

    # Generate Y
    Y = self.GenerateY(X, U, Z)

    # Generate M
    M = self.GenerateM(X, U, Y, single = self.Single)

    return X, Z, U, Y, M, S

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