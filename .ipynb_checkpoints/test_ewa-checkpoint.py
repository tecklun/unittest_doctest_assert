import ewa
import unittest
import numpy as np
import pandas as pd

class Test_exponential_running_standardize(unittest.TestCase):
    """ 
    Test class for function ewa.exponential_weighted_average 
    Arguments: data, alpha=0.001,  eps=1e-4
    1. test_block_easy: single channel, 3 samples
    2. test_block_sgl: single channel (i.e. data.shape=(sample, 1))
    3. test_block_dbl: 2 channel (i.e. data.shape=(sample,2))
    4. test_block_dbl_float: 2 channel (i.e. data.shape=(sample,2)). Input is float
    """
    

    def test_block_easy(self):
        data, alpha = np.array([[1],[2],[3]]), 0.1   # Initialize
        actual = ewa.exponential_weighted_average(data, alpha)   # Actual
        
        # Expected
        expected = np.zeros((data.shape))
        expected[0,0] = data[0,0]
        for i in range(1, len(data)):
            expected[i,0] = alpha*data[i,0] + (1-alpha)*mean[i-1,0]

        np.testing.assert_array_equal(actual, expected)    # Check correct
        
    def test_block_sgl(self):
        """
        2. test_block_sgl: single channel (i.e. data.shape=(sample, 1))
        """
        import pandas as pd
        # Initialize
        data = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
        alpha = 0.1
        
        # Actual
        actual = ewa.exponential_weighted_average(data, alpha)
        
        # Expected
        mean = np.zeros((data.shape))
        mean[0,0] = data[0,0]
        for i in range(1, len(data)):
            mean[i,0] = alpha*data[i,0] + (1-alpha)*mean[i-1,0]
        
        expected = mean
        
        # Check correct
        np.testing.assert_array_equal(actual, expected)

        
    def test_block_dbl(self):
        """
        2. test_block_dbl: 2 channel (i.e. data.shape=(sample,2))
        """
        import pandas as pd
        # Initialize
        data = np.array([[1,10],[2,10],[3,10],[4,40],[5,20],[6,10],[7,10],[8,10],[9,10],[10,10]])
        alpha = 0.1
        
        # Actual
        actual = ewa.exponential_weighted_average(data, alpha)
        
        # Expected
        mean = np.zeros((data.shape))
        mean[0,:] = data[0,:]
        for i in range(1, len(data)):
            mean[i,:] = alpha*data[i,:] + (1-alpha)*mean[i-1,:]
        
        expected = mean
        
        # Check correct
        np.testing.assert_array_equal(actual, expected)
        
        
    def test_block_dbl_float(self):
        """
        3. test_block_dbl_float: 2 channel (i.e. data.shape=(sample,2)). Input is float
        """
        import pandas as pd
        # Initialize
        data = np.array([[1.,10.],[2.,10.],[3.,10.],[4.,40.],[5.,20.],[6.,10.],[7.,10.],[8.,10.],[9.,10.],[10.,10.]])
        alpha = 0.1
        
        # Actual
        actual = ewa.exponential_weighted_average(data, alpha)
        
        # Expected
        mean = np.zeros((data.shape))
        mean[0,:] = data[0,:]
        for i in range(1, len(data)):
            mean[i,:] = alpha*data[i,:] + (1-alpha)*mean[i-1,:]
    
        expected = mean
        
        # Check correct
        np.testing.assert_array_equal(actual, expected)
        
if __name__ == '__main__':
    unittest.main(exit=False)
