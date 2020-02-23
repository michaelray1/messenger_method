"""Module containing messenger method code that has been packaged in a more convenient manner"""

import numpy as np
import healpy as hp

class Mmwf:


    def __init__(self, N_cov, Sig_cov, Cooling):
        """
        Wiener filter object. Contains methods to filter using pure-B messenger method. In the future, we may add pure-E and pure-T options.
        
        Parameters
        N_cov - Noise covariance object created with use of Noise_cov class below
        Sig_cov - Signal covariance object created with the use of Sig_cov class below
        Cooling - Cooling schedule object created with the use of Cooling class below"""
        
        """Check for consistency between sizes of noise and signal covariance"""

        self.N_cov = N_cov
        self.Sig_cov = Sig_cov
        self.Cooling = Cooling
        self.Nbar = N_cov.Nbar


    def solve_pixeqn(self, lam, data_qu, s):
        """Solves the first messenger method equation, which is done in the pixel domain. You must provide lambda, QU input data, and the current signal reconstruction in pixel space."""
        if self.N_cov.is_diagonal == False:
            t = np.matmul(mat_inverse(self.N_cov.Nbar_inverse() + self.N_cov.lamTpix_inverse(lam)), self.N_cov.Nbarinv_times(data_qu) + self.N_cov.invTpix_times(lam, s))
        else:
            t = (self.N_cov.Nbar_inverse() + self.N_cov.lamTpix_inverse(lam)) ** (-1) * (self.N_cov.Nbar_inverse() * data_qu + self.N_cov.lamTpix_inverse(lam) * s)
        return t

    def solve_spheqn(self, lam, tsph):
        """Solves the second messenger method equation, which is done in the spherical harmonic domain. You must provide lambda and the current messenger field vector in spherical harmonic space."""
        if self.Sig_cov.is_diagonal == False:
            sig = np.matmul(mat_inverse(self.Sig_cov.pseudo_inv() + self.N_cov.lamTsph_inverse(lam), self.N_cov.invTsph_times(lam, tsph)))
        else:
            sig = (self.Sig_cov.inverse() + self.N_cov.lamTsph_inverse(lam)) ** (-1) * self.N_cov.lamTsph_inverse(lam) * tsph
        return sig

    def mat_inverse(self, matrix):
        """Computes the inverse of the given matrix"""
        if len(np.shape(matrix)) == 1:
            inv = matrix**(-1)
        else:
            inv = np.linalg.inv(matrix)
        return inv


    def do_iteration(self, lam, s, data):
        """Performs one iteration of the messenger method algorithm.

        Parameters
        lam - Give a scalar that represents lambda in the messenger method equations
        s - Give a signal that represents s in the pixel domain. This should be size Npix * 2
        data - Give a numpy array of size 3 by Npix. This is the I,Q,U map being filtered and data should be in the order of I,Q,U."""
        
        data_qu = np.concatenate((data[0,:], data[1,:]), axis=0)

        tpix = self.solve_pixeqn(lam = lam, data_qu = data_qu, s = s)
        tpix_q = tpix[:self.N_cov.Npix]
        tpix_u = tpix[self.N_cov.Npix:]
        t_e_b = hp.map2alm((data[0,:], tpix_q, tpix_u), lmax=self.N_cov.ellmax, pol=True)
        tsph_e = t_e_b[1]
        tsph_b = t_e_b[2]
        tsph = np.concatenate((tsph_e, tsph_b), axis = 0)
        
        ssph = self.solve_spheqn(lam = lam, tsph = tsph)
        ssph_e = ssph[:self.N_cov.Nsph]
        ssph_b = ssph[self.N_cov.Nsph:]
        weiner_iqu = hp.alm2map((t_e_b[0], ssph_e, ssph_b), nside = self.N_cov.Nside, lmax = self.N_cov.ellmax, pol=True)
        s = np.concatenate((weiner_iqu[1], weiner_iqu[2]), axis = 0)
                            
        return s 


    def filter_map_pureB(self, data):
        """This function calls the do_iteration function with different values of lambda, while updating the signal reconstruction at each iteration. The end result is a wiener filtered set of alm's.
        
        Parameters
        data - Give the map that you want to filter. This will be a numpy array of size 3 by Npix. Data should be in the order of I,Q,U."""
        s = np.zeros(self.N_cov.Npix * 2)
        for lam in self.Cooling.lam_list:
            s = self.do_iteration(lam, s, data)

        """Transform s from pixel basis to spherical harmonic domain"""
        ssph = hp.map2alm((data[0,:], s[:self.N_cov.Npix], s[self.N_cov.Npix:]), lmax = self.N_cov.ellmax, pol = True)
        ssph_eb = np.concatenate((ssph[0,:], ssph[1,:]), axis=0)

        if self.N_cov.is_diagonal == False:
            s_final = np.matmul(np.matmul(self.Sig_cov.S, self.Sig_cov.pseudo_inv()), ssph_eb)
        else:
            s_final = self.Sig_cov.S * self.Sig_cov.pseudo_inv() * ssph_eb

        return s_final
        


class Noise_cov:
    """Noise covariance object.

    Parameters
    Nside - Noise covariance matrix as a numpy array. It will be size NPIX by NPIX, or size NPIX by one if it is diagonal.
    """


    def __init__(self, Nside):
        """Initializes noise covariance object.
        
        Parameters
        Nside - Give an integer that is the nside of your map.
        """
        self.Nside = Nside
        self.Npix = 12*Nside**2
        self.ellmax = 3*self.Nside - 1
        ells = np.arange(self.ellmax+1)
        self.Nsph = np.sum(ells+1)


    def make_matrix(self, matrix = None):
        """Sets Noise_cov.N to be whatever is given as matrix in the input. Noise_cov.N is the full noise covariance matrix. This also fixes the T covariance matrix (in both the pixel and spherical harmonic domain) and the Nbar covariance matrix.

        Parameters
        matrix - Give the noise covariance matrix you want to use. Size of matrix should be Npix*2 by Npix*2 if Noise_cov.is_diagonal is False. If Noise_cov.is_diagonal is True, then matrix should be size Npix*2
        """
        if len(np.shape(matrix)) == 1:
            self.is_diagonal = True
        else:
            self.is_diagonal = False

        self.N = matrix

        if self.is_diagonal == False:
            self.tau = np.min(np.diagonal(self.N))
            self.T_pix = np.identity(self.Npix * 2) * self.tau
            self.Nbar = self.N - self.T_pix
            self.T_sph = T_pix * 4 * np.pi / self.Npix
        else:
            self.tau = np.min(self.N)
            self.T_pix = np.ones(self.Npix*2) * self.tau
            self.Nbar = self.N - self.T_pix
            self.T_sph = self.tau * np.ones(self.Nsph * 2) * (4.0 * np.pi) / self.Npix


    def N_inverse(self):
        """Returns the inverse of the noise covariance matrix
        """
        if self.is_diagonal == False:
            inv = np.linalg.inv(self.N)
        else:
            inv = self.N ** (-1)
        return inv


    def Nbar_inverse(self):
        """Computes the inverse of the Nbar matrix
        """
        if self.is_diagonal == False:
            inv = np.linalg.inv(self.Nbar)
        else:
            inv = self.Nbar ** (-1)
        return inv


    def T_inverse(self):
        """Computes the inverse of the T covariance matrix
        """
        if self.is_diagonal == False:
            inv = np.linalg.inv(self.T)
        else:
            inv = self.T ** (-1)
        return T


    def lamTpix_inverse(self, lam):
        """Computes the inverse of lambda times the T_pix matrix
        
        Parameters
        lam - Give a scalar quantity"""
        if self.is_diagonal == False:
            inv = np.linalg.inv(lam * self.T_pix)
        else:
            inv = (lam * self.T_pix) ** (-1)
        return inv


    def lamTsph_inverse(self, lam):
        """Computes the inverse of lambda times the T_sph matrix
        
        Parameters
        lam - Give a scalar quantity"""
        if self.is_diagonal == False:
            inv = np.linalg.inv(lam * self.T_sph)
        else:
            inv = (lam * self.T_sph) ** (-1)
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


    def __init__(self):
        """Initializes signal covariance object.
        
        Parameters
        S - Signal covariance matrix as a numpy array. It should be Nsph*2 by Nsph*2 if it is dense. It should be just Nsph*2 if it is diagonal.
        """
        pass

    def make_matrix(self, matrix):
        """Creates the signal covariance matrix and saves it as Sig_cov.S

        Parameters
        matrix - Give the signal covariance matrix. This is either Nsph*2 by Nsph*2 if the matrix is not diagonal, or size Nsph*2 if it is diagonal.
        """
        if len(np.shape(matrix)) == 1:
            self.is_diagonal = True
            self.Nsph = int(len(matrix)/2)
        else:
            self.is_diagonal = False
            self.Nsph = int(len(np.diagonal(matrix))/2)
        
        self.S = matrix


    def inverse(self):
        """Computes the inverse of the signal covariance matrix
        """
        if self.is_diagonal == False:
            inv = np.linalg.inv(self.S)
        else:
            inv = self.S ** (-1)
        return inv


    def pseudo_inv(self):
        """Returns the inverse of the first Nsph entries in the signal covariance and sets the last Nsph elements to zero. This is for a pure B estimator. Thus, it is set up so the first Nsph entries in Sig_cov.S are the E mode components and the last Nsph entries are the B mode components.
        """
        if self.is_diagonal == False:
            raise ValueError('Signal covariance is not diagonal, pseudo inverse not supported.')
        else:
            inv_top = self.S[:self.Nsph] * 0.0
            inv_bottom = self.S[self.Nsph:]**(-1)
            inv = np.concatenate((inv_top, inv_bottom), axis = 0)
        return inv

    

class Cooling:
    """Cooling object. This is how you construct a cooling schedule for lambda in the iterating process.
    """

    
    def __init__(self, lam_list = None):
        """Initializes Cooling object. Cooling object holds the information about how to adjust lambda when iterating.
        
        Parameters
        lam_list - Give a numpy array that is the list of lambda values you want to feed through the messenger method algorithm. The final value should be one, since at lambda equals one, the messenger method equations reduce to the wiener filter.
        """
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
