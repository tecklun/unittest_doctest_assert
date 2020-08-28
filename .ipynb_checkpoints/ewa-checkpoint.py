import pandas as pd
import numpy as np


def exponential_weighted_average(data, alpha=0.001):
    """
    Function: Using exponential weighted average to perform normalization of data.
    The exponential weighted average is calculated recursively given:
        y<0> = x<0>
        y<t> = (1 - alpha)*y<t-1> + alpha*x<t>
    
    Arguments:
    data: numpy array of shape (samples, channels)
    alpha: float. value of alpha as in equation above

    Returns:
    normalized: numpy array of shape (samples, channels). Which is the normalized version of input data
    >>> import numpy as np
    >>> data = np.array([[1],[2],[3]])
    >>> alpha = 0.1
    >>> exponential_weighted_average(data, alpha, eps)
    array([[1.  ],
           [1.1 ],
           [1.29]])
    """
    # Check validity of input
    assert_check_ewa(data, alpha)
    
    df = pd.DataFrame(data)
    mean = df.ewm(alpha=alpha, adjust=False).mean()  
    mean = np.array(mean)
    return mean

def assert_check_ewa(data, alpha):
    # Check validity of data 
    # type
    assert type(data)==np.ndarray, 'data must be type numpy.ndarray'
    
    # shape 
    assert len(data.shape)==2, 'data must have shape (time, ch)'  
    
    # Check validity of alpha
    # type
    assert type(alpha) == float, 'alpha must be of type float'    


if __name__ == '__main__':
    import doctest
    doctest.testmod()