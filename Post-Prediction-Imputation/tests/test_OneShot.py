import numpy as np
import sys
import pandas as pd

sys.path.append('../')
from OneShot import OneShotTest
import numpy as np
import pandas as pd
import unittest
from OneShot import OneShotTest

class TestOneShotTest(unittest.TestCase):

    def setUp(self):
        self.ost = OneShotTest(N=10, Single=False)

    def test_to_csv(self):
        X = np.array([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5]])
        assert X.shape == (5, 5)
        Z = np.array([[0],[0],[0],[0],[0]])
        assert Z.shape == (5, 1)
        M = np.array([[0],[0],[1],[1],[1]])
        assert M.shape == (5, 1)
        Y = np.array([[1.0],[2.0],[3.0],[4.0],[5.0]])
        assert Y.shape == (5, 1)
        Y_masked = np.ma.masked_array(Y, mask=M)
        Y_masked = Y_masked.filled(np.nan)
        assert Y_masked.shape == (5, 1)
        assert Y_masked[0] == np.array([1.0])
        assert Y_masked[1] == np.array([2.0])
        assert np.isnan(Y_masked[2][0]) == True
        assert np.isnan(Y_masked[3][0]) == True
        assert np.isnan(Y_masked[4][0]) == True
        expected_df1 = pd.DataFrame({
            0: [0, 0, 0, 0, 0],
            1: [1, 1, 1, 1, 1],
            2: [2, 2, 2, 2, 2],
            3: [3, 3, 3, 3, 3],
            4: [4, 4, 4, 4, 4],
            5: [5, 5, 5, 5, 5],
            6: [1.0, 2.0, 3.0, 4.0, 5.0],
            
        }, dtype=np.float64)

        expected_df2 = pd.DataFrame({
            0: [0, 0, 0, 0, 0],
            1: [1, 1, 1, 1, 1],
            2: [2, 2, 2, 2, 2],
            3: [3, 3, 3, 3, 3],
            4: [4, 4, 4, 4, 4],
            5: [5, 5, 5, 5, 5],
            6: [1.0, 2.0, np.nan, np.nan, np.nan]
        }, dtype=np.float64)
        df1 = pd.DataFrame(np.concatenate((Z, X, Y), axis=1))
        # assert that the second resulting dataframe has the expected values

        pd.testing.assert_frame_equal(df1, expected_df1)
        df2 = pd.DataFrame(np.concatenate((Z, X, Y_masked), axis=1))   

        pd.testing.assert_frame_equal(df2, expected_df2)            
        
    def test_split_df(self):
        # create a DataFrame with known values
        ost = OneShotTest(N=10, Single=False)

        data = pd.DataFrame({ 'Z': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            'X1': [1, 2, 3, 4, 5, 1 , 1, 1, 1, 1],
                             'X2': [6, 7, 8, 9, 10, 1 , 1, 1, 1, 1],
                             'X3': [11, 12, 13, 14, 15, 1 , 1, 1, 1, 1],
                             'X4': [16, 17, 18, 19, 20, 1 , 1, 1, 1, 1],
                             'X5': [21, 22, 23, 24, 25, 1 , 1, 1, 1, 1],
                             'Y': [26, 27, 28, 29, 30, 1 , 1, 1, 1, 1],
                             })
        # split the dataset using the split_df function
        df1, df2 = self.ost.split_df(data)
        
        # assert that the first resulting dataframe has the expected values
        expected_df1 = pd.DataFrame({
            'Z': [0, 0, 0, 0, 0],
            'X1': [1, 2, 3, 4, 5],
            'X2': [6, 7, 8, 9, 10],
            'X3': [11, 12, 13, 14, 15],
            'X4': [16, 17, 18, 19, 20],
            'X5': [21, 22, 23, 24, 25],
            'Y': [26, 27, 28, 29, 30],
        })
        pd.testing.assert_frame_equal(df1, expected_df1)
        
        # assert that the second resulting dataframe has the expected values
        expected_df2 = pd.DataFrame({
            'Z': [ 0, 0, 0, 0, 0],
            'X1': [ 1 , 1, 1, 1, 1],
            'X2': [ 1 , 1, 1, 1, 1],
            'X3': [ 1 , 1, 1, 1, 1],
            'X4': [ 1 , 1, 1, 1, 1],
            'X5': [ 1 , 1, 1, 1, 1],
            'Y': [ 1 , 1, 1, 1, 1],
        })
        pd.testing.assert_frame_equal(df2, expected_df2)

    def test_holm_bonferroni(self):
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        result = self.ost.holm_bonferroni(p_values)
        self.assertIsInstance(result, bool)
        assert result == True

        p_values = [0.01, 0.05]
        result = self.ost.holm_bonferroni(p_values)
        self.assertIsInstance(result, bool)
        assert result == True

        p_values = [0.1, 0.05]
        result = self.ost.holm_bonferroni(p_values)
        self.assertIsInstance(result, bool)
        assert result == False        

    def test_T(self):
        z = np.array([1, 1, 0, 0, 1])
        y = np.array([5, 3, 4, 2, 1])
        result = self.ost.T(z, y)
        assert result == 9
    
    def test_getT(self):
        # Prepare a small sample dataframe
        data = {
            'Z': [0, 1, 1, 0, 1],
            'X1': [1, 2, 3, 4, 5],
            'X2': [5, 4, 3, 2, 1],
            'X3': [2, 3, 4, 5, 6],
            'X4': [6, 5, 4, 3, 2],
            'X5': [3, 4, 5, 6, 7],
            'Y': [10, 20, 30, 40, 50]
        }
        df = pd.DataFrame(data)

        # Set parameters for the getT method
        G = None
        indexY = 6
        lenY = 1

        # Call getT with the given parameters
        t = self.ost.getT(G, df, indexY, lenY)

        # Check the correctness of the output
        self.assertIsNotNone(t)
        self.assertEqual(len(t), lenY)
        assert len(t) == lenY
        assert t[0] == 10
    
    def test_worker(self):
        G1 = None
        G2 = None
        X = np.random.randn(200, 3)
        Y = np.random.randn(200, 3)
        Y_masked = np.random.randn(200, 3)
        t1_obs = np.random.randn(3)
        t2_obs = np.random.randn(3)
        L = 100
        verbose = False
        args = (X, Y, Y_masked, G1, G2, t1_obs, t2_obs, L, verbose)
        result = self.ost.worker(args)
        self.assertIsInstance(result, tuple)

    def test_one_shot_test_parallel(self):
        Z = np.random.randint(2, size=(200, 1))
        X = np.random.randn(200, 3)
        M = np.random.randint(2, size=(200, 3))
        Y = np.random.randn(200, 3)
        G1 = None
        G2 = None
        L = 1000
        verbose = False
        result = self.ost.one_shot_test_parallel(Z, X, M, Y, G1, G2, L=L, verbose=verbose)
        self.assertIsInstance(result, tuple)

    def test_one_shot_test(self):
        Z = np.random.randint(2, size=(200, 1))
        X = np.random.randn(200, 3)
        M = np.random.randint(2, size=(200, 3))
        Y = np.random.randn(200, 3)
        G1 = None
        G2 = None
        L = 1000
        verbose = False
        result = self.ost.one_shot_test(Z, X, M, Y, G1, G2, L=L, verbose=verbose)
        self.assertIsInstance(result, tuple)

