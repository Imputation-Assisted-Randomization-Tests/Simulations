import numpy as np
import sys
sys.path.append('../')
from Simulation import DataGenerator

# Constants for testing
N = 100
N_S = 5



def test_single_unobserved():
    DataGen = DataGenerator(N = N, N_T = int(N_S / 2), N_S = int(N_S / 2), beta_11 = 1, beta_12 = 1, beta_21 = 1, beta_22 = 1, beta_23 = 1, beta_31 = 1, beta_32 = 0, MaskRate=0.5,Unobserved=1, Single=1, verbose=0)
    X, Z, U, Y, M, S = DataGen.GenerateData()
    assert int(M.sum()) == 50
    assert M.shape == (N, 3)
    assert Y.shape == (N, 1)
    assert X.shape == (N, 5)
    assert U.shape == (N, 1)
    assert Z.shape == (N, 1)
    assert S.shape == (N, 1)

def test_single_observed():
    DataGen = DataGenerator(N = N, N_T = int(N_S / 2), N_S = int(N_S / 2), beta_11 = 1, beta_12 = 1, beta_21 = 1, beta_22 = 1, beta_23 = 1, beta_31 = 1, beta_32 = 0, MaskRate=0.5,Unobserved=0, Single=1, verbose=0)
    X, Z, U, Y, M, S = DataGen.GenerateData()
    assert int(M.sum()) == 50
    assert M.shape == (N, 1)
    assert Y.shape == (N, 1)
    assert X.shape == (N, 5)
    assert U.shape == (N, 1)
    assert Z.shape == (N, 1)
    assert S.shape == (N, 1)

def test_multiple_unobserved():
    DataGen = DataGenerator(N = N, N_T = int(N_S / 2), N_S = int(N_S / 2), beta_11 = 1, beta_12 = 1, beta_21 = 1, beta_22 = 1, beta_23 = 1, beta_31 = 1, beta_32 = 0, MaskRate=0.5,Unobserved=1, Single=0, verbose=0)
    X, Z, U, Y, M, S = DataGen.GenerateData()
    assert int(M.sum()) == 150
    assert M.shape == (N, 3)
    assert Y.shape == (N, 3)
    assert X.shape == (N, 5)
    assert U.shape == (N, 1)
    assert Z.shape == (N, 1)
    assert S.shape == (N, 1)

def test_multiple_observed():
    DataGen = DataGenerator(N = N, N_T = int(N_S / 2), N_S = int(N_S / 2), beta_11 = 1, beta_12 = 1, beta_21 = 1, beta_22 = 1, beta_23 = 1, beta_31 = 1, beta_32 = 0, MaskRate=0.5,Unobserved=0, Single=0, verbose=0)
    X, Z, U, Y, M, S = DataGen.GenerateData()
    assert int(M.sum()) == 150
    assert M.shape == (N, 3)
    assert Y.shape == (N, 3)
    assert X.shape == (N, 5)
    assert U.shape == (N, 1)
    assert Z.shape == (N, 1)
    assert S.shape == (N, 1)



