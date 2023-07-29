import numpy as np
from mv_laplace import MvLaplaceSampler
import pandas as pd
from scipy.stats import logistic
from random import sample


class DataGenerator:
  def __init__(self,*, N = 1000, strata_size = 10, beta_11 = 1, beta_12 = 1, beta_21 = 1, beta_22 = 1, beta_23 = 1, beta_31 = 1,beta_32 = 1, MaskRate = 0.3, Unobserved = True, Single = True, linear_method = 0, verbose = False, bias = False, Missing_lambda):
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
    self.linear_method = linear_method
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
    print(Z)

    print("****************")
    Z = []
    for i in range(self.totalStrataNumber):
      Z.append(np.random.binomial(1, 0.5, self.strata_size))
    Z = np.concatenate(Z).reshape(-1, 1)
    print(Z)
    return Z


  def GenerateIndividualEps(self):
      mean = 0
      std = 0.2
      eps = np.random.normal(mean, std, self.N)
      eps = eps.reshape(-1,)

      return eps
  
  def GenerateStrataEps(self):
      eps = []

      for i in range(self.totalStrataNumber):
          eps.append(np.full(self.strata_size, np.random.normal(0, 0.1)))

      eps = np.concatenate(eps).reshape(-1,)
      return eps
  
  def GenerateXInter(self, X):
      biases = []

      for i in range(self.totalStrataNumber):
          strata = X[i * self.strata_size : (i+1) * self.strata_size, 0]  # select the first column in the strata
          biases.append(np.full(self.strata_size, np.mean(strata)))

      biases = np.concatenate(biases).reshape(-1,)
      return biases

  def GenerateYInter(self, Y):
      biases = []

      for i in range(self.totalStrataNumber):
          strata = Y[i * self.strata_size : (i+1) * self.strata_size, 0]  # select the first column in the strata
          biases.append(np.full(self.strata_size,1/2 * np.mean(strata)))

      biases = np.concatenate(biases).reshape(-1,)

      return biases

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

    #def sum8(): 
    sum8 = np.zeros(self.N)
    for p in range(1,6):
      for p_2 in range(1,6):
        for p_3 in range(1,6):
          sum8 += X[:,p-1] * X[:,p_2-1] * np.cos(1 - 4*X[:,p_3-1])
    sum8 = (1.0  / np.sqrt(5 * 5 )) * sum8  

    U = U.reshape(-1,)
    Z = Z.reshape(-1,)


    Y_n3 = self.beta_32 * Z +  self.beta_22 * Z * X[:,0]+ self.beta_12 * Z * sum5 + sum3 + sum4 +  StrataEps+ IndividualEps 
      
    Y = Y_n3.reshape(-1, 1)

    return Y


  def GenerateM(self, X, U, Y, XInter, YInter, single = True):
      
      U = U.reshape(-1,)
      n = X.shape[0]
      if self.verbose:
        M_lamda_Y = np.zeros((n, 1))
        M_lamda_X = np.zeros((n, 1))
        M_lamda_U = np.zeros((n, 1))
        M_lamda_XInter= np.zeros((n, 1))
        M_lamda_YInter= np.zeros((n, 1))

      if single:
        M = np.zeros((n, 1))
        M_lamda = np.zeros((n, 1))
        for i in range(n):
          sum3 = 0
          for p in range(1,6):
              sum3 += p * X[i,p-1] 
          sum3 = (1.0  / np.sqrt(5)) * sum3

          sum2 = 0
          for p in range(1,6):
            sum2 += p * np.cos(X[i,p-1])
          sum2 =  (1.0  / np.sqrt(5)) * sum2


        M_lamda[i][0] = sum3 + sum2  + 10 * logistic.cdf(Y[i, 0]) + U[i] + XInter[i] + YInter[i]

        lambda1 = np.percentile(M_lamda, 100 * (1-self.MaskRate)) #self.Missing_lambda #

        for i in range(n):
          sum3 = 0
          for p in range(1,6):
              sum3 += p * X[i,p-1] 
          sum3 = (1.0  / np.sqrt(5)) * sum3

          sum2 = 0
          for p in range(1,6):
            sum2 += p * np.cos(X[i,p-1])
          sum2 =  (1.0  / np.sqrt(5)) * sum2

          if self.verbose:
            M_lamda_Y[i][0] = 10 * logistic.cdf(Y[i, 0])
            M_lamda_X[i][0] = sum2 + sum3
            M_lamda_U[i][0] = U[i]
            M_lamda_XInter[i][0] = XInter[i]
            M_lamda_YInter[i][0] = YInter[i]


          if sum3 + sum2  + 10 * logistic.cdf(Y[i, 0])> lambda1:
            M[i][0] = 1 

        if self.verbose:
          data = pd.DataFrame(M_lamda_Y, columns=['Y1'])
          data['X'] = M_lamda_X[:,0]
          data['U'] = M_lamda_U[:,0]
          data['Xinter'] = M_lamda_XInter[:,0]
          data['Yinter'] = M_lamda_YInter[:,0]
          print(data.describe())
          print(pd.DataFrame(M).describe())

        #with open('lambda.txt', 'a') as f:
        #  f.write(str(lambda1) + '\n')

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

            M_lamda[i][1] = (1.0  / np.sqrt(5))*((X[i, :]**3).sum() + sum2  + -1 * Y[i, 0] + -1 * Y[i, 1])

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

            values[0] = (1.0  / np.sqrt(5))* logistic.cdf(X[i, :]).sum() + sum1 + -1 * logistic.cdf(Y[i, 0])
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

    # generate Interf
    StrataEps = self.GenerateStrataEps()
    IndividualEps = self.GenerateIndividualEps()
    XInter = self.GenerateXInter(X)
    

    # Generate Y
    Y = self.GenerateY(X, U, Z, StrataEps, IndividualEps)
    YInter = self.GenerateYInter(Y)

    # Generate M
    M = self.GenerateM(X, U, Y, XInter, YInter, single = self.Single)

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