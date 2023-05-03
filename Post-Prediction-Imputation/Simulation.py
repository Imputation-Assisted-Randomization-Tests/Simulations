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

    # Calculate Y_n1
    Y_n1 = (self.beta_11 * Z + self.beta_12 * Z * sum1  + sum2) 

    # Compute Yn2
    Y_n2 = (self.beta_21 * Z + self.beta_22 * Z * X[:,0] + sum3 + sum4) 

    # Compute Yn3
    Y_n3 = (self.beta_31 * Z + sum6) 
    
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
        
        for i in range(n):
          if np.random.uniform() < self.MaskRate:
            M[i][0] = 1
          else:
            M[i][0] = 0
        return M
        
      else:
        M = np.zeros((n, 3))
       #make M randomly generated based on self.MaskRate
        for i in range(n):
          if np.random.uniform() < self.MaskRate:
            M[i][0] = 0
          else:
            M[i][0] = 0

          if np.random.uniform() < self.MaskRate:
            M[i][1] = 0
          else:
            M[i][1] = 0

          if np.random.uniform() < self.MaskRate:
            M[i][2] = 1
          else:
            M[i][2] = 0
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