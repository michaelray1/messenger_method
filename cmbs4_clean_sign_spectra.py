import os
os.chdir('/users/PES0740/ucn3066/messenger')
import healpy as hp
import numpy as np
import sys
import messenger as msg

NSIDE = 512
NPIX = 12*NSIDE**2
ELLMAX = 3*NSIDE-1
NSPH = int(np.sum(1.0 * np.arange(ELLMAX+1) + 1.0))

mask = hp.read_map('512.fits')
mask_nans = np.where(np.isnan(mask))
mask[np.isnan(mask)] = 1e-10

cls = np.genfromtxt("/fs/project/PES0740/sky_yy/cmb/cls/ffp10_lensedCls.dat")
cls = np.insert(cls, [0], [1,0,0,0,0], axis = 0)
cls = np.insert(cls, [0], [0,0,0,0,0], axis = 0)
c_ell = np.array([cls[:,0]])
c_spectra = cls[:,1:]
c_ells = np.concatenate((np.transpose(c_ell), c_spectra), axis = 1)
c_ells_t = np.transpose(c_spectra)
B_pspec_cov = c_ells_t[3]
input_sup_BB = c_ells_t[3]

os.chdir('/users/PES0740/ucn3066/messenger/CMBS4/Sig_plus_N')
d_elltwid = np.load('All Sign plus Noise Output D_ell.npz')['arr_0'][:,:370]
binned_delltwid = np.empty([10,100])
for i in range(100):
    binned_delltwid[:,i] = msg.bin_pspec(d_elltwid[i,:],20,370,35)

os.chdir('/users/PES0740/ucn3066/messenger/CMBS4/Noise')
binned_noisebias = np.load('noise_bias_ellmin20_ellmax370_binsize35_dl.npz')['arr_0']

os.chdir('/users/PES0740/ucn3066/messenger/CMBS4/bpwf/BB2BB')
binned_sup_facs = np.load('take2_summed_bpwf.npz')['arr_0']

d_ell_binned = np.empty([10,100])
for j in range(100):
    for i in range(10):
        d_ell_binned[i,j] = (binned_delltwid[i,j] - binned_noisebias[i])/binned_sup_facs[i]

os.chdir('/users/PES0740/ucn3066/messenger/CMBS4/Sig_plus_N')
np.savez('Cleaned Signal plus Noise Spectra Using Binned d_elltwid.npz',d_ell_binned)
