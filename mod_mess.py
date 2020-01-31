"""Module containing messenger method code that has been packaged in a more convenient manner"""

import numpy as np
import healpy as hp

class mmwf:
    """wiener filter by messenger method object. 

    Parameters:
    N - Noise covariance object.
    S - Signal covariance object
    T - Messenger field covariance object
    cooling - Cooling schedule object
    """

    def __init__(self, Nbar, S, T, data, cooling):
        self.Nbar = Nbar
        self.S = S
        self.T = T
        self.data = data
        self.cooling = cooling

    def mat_inverse(self, matrix):
        """Computes the inverse of the given matrix"""
        inv = np.linalg.inverse(matrix)
        return inv

    def do_iteration(self, t, lam, s):
        tpix = ((self.Nbar.inverse().times(self.data)) + (self.T.lam_inverse(lam).times(s))) * mat_inverse((self.Nbar.inverse() + (self.T.lam_inverse(lam))))

    def filter_map(self):
        for lam in self.cooling:
            self.do_iteration(self.Nbar, self.S, self.T, self.t, self.data, lam)


class N:
    """Noise covariance object.

    Parameters
    cov_mat - Noise covariance matrix as a numpy array. It will be size NPIX by NPIX, or size NPIX by one if it is diagonal.
    """

    def __init__(self, Npix):
        self.Npix = Npix
        self.Nside = np.sqrt(Npix/12)
        self.ellmax = 3*self.Nside**2
        ells = np.arange(self.ellmax)                                                                                                                       
        self.Nsph = np.sum(ells+1)                                                                                                             
        self.cov_mat = np.ones((Npix*2, Npix*2), dtype = np.float64)

    def make_matrix(self, matrix):
        """Sets N.cov_mat to be whatever is given as matrix in the input

        Parameters
        matrix - Give the noise covariance matrix you want to use. Size of matrix should be Npix*2 by Npix*2
        """
        self.cov_mat = matrix

    def inverse(self):
        """Returns the inverse of the noise covariance matrix
        """
        inv = np.linalg.inv(self.cov_mat)
        return inv

    def times(self, x):
        """Returns the matrix product of the noise covariance matrix with the given matrix x"""
        return np.matmul(self.cov_mat, x)


class T:
    """Messenger field covariance object.

    Parameters
    N - Noise covariance object
    """
    def __init__(self, N):
        self.Npix = N.Npix
        self.Nsph = N.Nsph                                                                                                                       
        self.cov_mat_pix = min(N.cov_mat.diagonal()) * np.identity(N.Npix*2)
        self.cov_mat_sph = min(N.cov_mat.diagonal()) * np.identity(N.Nsph*2) * (4.0*np.pi) / N.Npix

    def times_lam(self, lam):
        """Multiplies T matrix by scalar value lambda"""
        product = lam * self.cov_mat_pix
        return product

    def lam_inverse(self, lam):
        """Returns the inverse of T times lambda"""
        inv = np.linalg.inv(times_lam(lam))
        return inv

    def times(self, x):
        """Returns the matrix product of the messenger covariance matrix with the given matrix x"""
        return np.matmul(self.cov_mat_pix, x)



class Nbar:
    """Nbar object.
    
    Parameters
    N - Noise covariance object
    T - Messenger field covariance object
    """

    def __init__(self, N, T):
        self.cov_mat = N.cov_mat - T.cov_mat


    def inverse(self):
        inv = np.linalg.inv(self.cov_mat)
        return inv

    def times(self, x):
        """Returns the matrix product of the Nbar covariance matrix with the given matrix x"""
        return np.matmul(self.cov_mat, x)


class S:
    """Signal covariance object.

    Parameters
    cov_mat - Signal covariance matrix as a numpy array. It will be size NPIX by NPIX, or size NPIX by one if it is diagonal.
    """

    def __init__(self, cov_mat):
        self.cov_mat = cov_mat

class cooling:
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
