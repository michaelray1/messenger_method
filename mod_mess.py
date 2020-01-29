"""Module containing messenger method code that has been packaged in a more convenient manner"""

import numpy as np
import healpy as hp

class mmwf(N, S, T, cooling):
    """wiener filter by messenger method object. 

    Parameters:
    N - Noise covariance object.
    S - Signal covariance object
    T - Messenger field covariance object
    cooling - Cooling schedule object
    """

    def __init__(self):
        mmwf.N = N
        mmwf.S = S
        mmwf.T = T
        mmwf.cooling = cooling

    def iteration(self, N, S, T, t, lam):


class N():
    """Noise covariance object.

    Parameters
    cov_mat - Noise covariance matrix as a numpy array. It will be size NPIX by NPIX, or size NPIX by one if it is diagonal.
    """

    def __init__(self, cov_mat):
        N.cov_mat = cov_mat

    def inverse(self):
        """Returns the inverse of the noise covariance matrix
        """
        inv = np.linalg.inv(self.cov_mat)
        return inv

    def times(self, x):
        """Returns the matrix product of the noise covariance matrix with the given matrix x"""
        
        


class S():
    """Signal covariance object.

    Parameters
    cov_mat - Signal covariance matrix as a numpy array. It will be size NPIX by NPIX, or size NPIX by one if it is diagonal.
    """

    def __init__(self, cov_mat):
        S.cov_mat = cov_mat

class cooling():
    """Cooling object. This is how you construct a cooling schedule for lambda in the iterating process.
    """

    
    def __init__(self, lam_list = None):
        self.lam_list = lam_list
        


    def standard_cooling(self, eta):
        """This function returns a list of lambda values as a numpy array that can be used in the iterative scheme.

        Parameters
        eta - Give a numeric value between zero and one. The function starts with lambda at 100 and multiplies lambda by eta on each iteration.
        """
        
        lam_list = []
        lam = 100
        while lam > 1:
            lam_list.append(lam)
            lam = lam*eta
        lam_list.append(1)
        lam_list = np.array(lam_list)
        return lam_list

    def linear_lam(self, maxlam, stepsize):
        """This function returns a list of lambda values as a numpy array that can be used in the iterative scheme.

        Parameters
        maxlam - Give a numeric value. This will be the first lambda value fed into the algorithm. 
        stepsize - Give a numberic value. Lambda will be decreased by this amount on each iteration.
        """
        
        lam_list = []
        lam = maxlam
        while lam > 1:
            lam_list.append(lam)
            lam = lam - stepsize
        lam_list.append(1)
        lam_list = np.array(lam_list)
        return lam_list
