"""Module containing messenger method code that has been packaged in a more convenient manner"""

import numpy as np
import healpy as hp

class Mmwf:
    """wiener filter by messenger method object. 

    Parameters:
    N - Noise covariance object.
    S - Signal covariance object
    T - Messenger field covariance object
    cooling - Cooling schedule object
    """

    def __init__(self, N_cov, Sig_cov, Cooling):
        self.N_cov = Noise_cov
        self.Sig_cov = Sig_cov
        self.cooling = Cooling
        self.T = N_cov.T
        self.Nbar = N_cov.Nbar

    def mat_inverse(self, matrix):
        """Computes the inverse of the given matrix"""
        inv = np.linalg.inverse(matrix)
        return inv


    def do_iteration(self, lam, s, data):
        """Performs one iteration of the messenger method algorithm.

        Parameters
        lam - Give a scalar that represents lambda in the messenger method equations
        s - Give a signal that represents s in the pixel domain. This should be size Npix * 2
        data - Give a vector that is 3 by Npix in size. This is the I,Q,U map being filtered and data should be in the order of I,Q,U."""
        
        data_qu = data[self.N.Npix:]
        def solve_pixeqn(self):
            t = np.matmul(mat_inverse(self.N_cov.Nbar_inverse() + self.N_cov.lamTpix_inverse(lam)), self.N_cov.Nbarinv_times(data_qu) + self.N_cov.invTpix_times(lam, s))
            return t
        
        def solve_spheqn(self, tsph):
            sig = np.matmul(mat_inverse(self.Sig_cov.pseudo_inv() + self.N_cov.lamTsph_inverse(lam), self.N_cov.invTsph_times(lam, tsph))
            return sig

        tpix = solve_pixeqn()
        tpix_q = tpix[:NPIX]
        tpix_u = tpix[NPIX:]
        t_e_b = hp.map2alm((data[0,:], tpix_q, tpix_u), lmax=ELLMAX, pol=True)
        tsph_e = t_e_b[1]
        tsph_b = t_e_b[2]
        tsph = np.concatenate((tsph_e, tsph_b), axis = 0)
        
        ssph = solve_spheqn(tsph)
        ssph_e = ssph[:self.N_cov.Nsph]
        ssph_b = ssph[self.N_cov.Nsph:]
        weiner_iqu = hp.alm2map((t_e_b[0], ssph_e, ssph_b), nside = self.N_cov.Nside, lmax = self.N_cov.ellmax, pol=True)
        s = np.concatenate((weiner_iqu[1], weiner_iqu[2]), axis = 0)
                            
        return s 


    def filter_map_pureB(self, data):
        s = np.zeros(self.N_cov.Npix * 2)
        for lam in self.Cooling.lam_list:
            s = self.do_iteration(lam, s, data)
        s_final = np.matmul(np.matmul(self.Sig_cov.S, self.Sig_cov.pseudo_inv()), s)


class Noise_cov:
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


    def make_matrix(self, matrix = None):
        """Sets N.cov_mat to be whatever is given as matrix in the input

        Parameters
        matrix - Give the noise covariance matrix you want to use. Size of matrix should be Npix*2 by Npix*2
        """
        self.N = matrix
        tau = np.min(np.diagonal(self.cov_mat))
        self.T_pix = np.identity(self.Npix * 2) * tau
        self.Nbar = self.N - self.T_pix
        self.T_sph = T_pix * 4 * np.pi / self.Npix


    def N_inverse(self):
        """Returns the inverse of the noise covariance matrix
        """
        inv = np.linalg.inv(self.cov_mat)
        return inv


    def Nbar_inverse(self):
        """Computes the inverse of the Nbar matrix
        """
        inv = np.linalg.inv(self.Nbar)
        return inv


    def T_inverse(self):
        """Computes the inverse of the T matrix
        """
        inv = np.linalg.inv(self.T)
        return T


    def lamTpix_inverse(self, lam):
        """Computes the inverse of lambda times the T_pix matrix
        
        Parameters
        lam - Give a scalar quantity"""
        inv = np.linalg.inv(lam * self.T_pix)
        return inv


    def lamTsph_inverse(self, lam):
        """Computes the inverse of lambda times the T_sph matrix
        
        Parameters
        lam - Give a scalar quantity"""
        inv = np.linalg.inv(lam * self.T_sph)
        return inv


    def Nbarinv_times(self, x):
        """Returns the matrix product of the noise covariance matrix with the given matrix x"""
        return np.matmul(self.Nbar_inverse(), x)


    def invTpix_times(self, lam, x):
        """Returns the matrix product of (lam * T_pix)**(-1) with the given matrix x.
        Give a scalar value for lam."""
        return np.matmul(lamTpix_inverse(lam), x)

    def invTsph_times(self, lam, x):
        """Returns the matrix product of (lam * T_sph)**(-1) with the given matrix x.
        Give a scalar for lam"""
        return np.matmul(lamTsph_inverse(lam), x)


class Sig_cov:
    """Signal covariance object.

    Parameters
    S - Signal covariance matrix as a numpy array. It will be size NPIX by NPIX, or size NPIX by one if it is diagonal.
    """

    def __init__(self, S):
        self.S = S


    def inverse(self):
        inv = np.linalg.inv(self.S)
        return inv


    def pseudo_inv(self):
        """Returns the inverse of the first Nsph entries in the signal covariance and sets the last Nsph elements to zero. This is for a pure B estimator. Thus, it is set up so the first Nsph entries in Sig_cov.S are the E mode components and the last Nsph entries are the B mode components.
        """
        inv_top = self.S[:Nsph] * 0.0
        inv_bottom = self.S[Nsph:]**(-1)
        inv = np.concatenate((inv_top, inv_bottom), axis = 0)
        return inv

    

class Cooling:
    """Cooling object. This is how you construct a cooling schedule for lambda in the iterating process.
    """

    
    def __init__(self, lam_list = None):
        self.lam_list = lam_list
        


    def standard_cooling(self, eta = 0.7):
        """This function returns a list of lambda values as a numpy array that can be used in the iterative scheme. Standard cooling begins with lambda at 1300, then decreases to 100 on the second iteration and from then on out multiplies by eta on each iteration. Once the value of lambda drops below 1, 5 lambda = 1 values are added to the list.

        Parameters
        eta - Give a numeric value between zero and one. The function starts with lambda at 100 and multiplies lambda by eta on each iteration.
        """
        
        lam_list = []
        lam_list.append(1300)
        lam = 100
        while lam > 1:
            lam_list.append(lam)
            lam = lam*eta
        i = 0
        while i < 5:
            lam_list.append(1)
            i += 1
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
