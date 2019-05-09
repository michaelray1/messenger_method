import os
import sys
import numpy as np
import healpy as hp
import messenger as msg

INDEX = int(sys.argv[1])
NSIMS = int(sys.argv[2])

cls = np.genfromtxt("/fs/project/PES0740/sky_yy/cmb/cls/ffp10_lensedCls.dat")
cls = np.insert(cls, [0], [1,0,0,0,0], axis = 0)
cls = np.insert(cls, [0], [0,0,0,0,0], axis = 0)
c_ell = np.array([cls[:,0]])
c_spectra = cls[:,1:]
c_ells = np.concatenate((np.transpose(c_ell), c_spectra), axis = 1)
c_ells_t = np.transpose(c_ells)
B_pspec_cov = c_ells_t[3,:1535]
Ncovmat = np.load('/users/PES0740/ucn3066/messenger/CMBS4/Noise/CMBS4 Ndiag.npz')['arr_0']
Ndiag = np.concatenate((Ncovmat[1,:],Ncovmat[2,:]),axis=0)

os.chdir('/users/PES0740/ucn3066/CMBS4_maps')
mask = hp.read_map('n0512.fits')
mask[np.isnan(mask)]=1e-10

NPIX = len(mask)
NSIDE = int(np.sqrt(NPIX/12))
ELLMAX = 3*NSIDE - 1
NSPH = int(np.sum(1.0 * np.arange(ELLMAX+1) + 1.0))
list_of_b_spectra1 = []
list_of_input_iqu = []
list_of_output_iqu = []
for x in range(NSIMS):
    i_q_u1 = hp.read_map('/users/PES0740/ucn3066/CMBS4_maps/cmbs4_04p00_comb_f095_b24_ellmin30_map_0512_mc_{:04d}.fits'.format(x+(10*INDEX)),field = (0,1,2))
    i_q_u1[np.isnan(i_q_u1)] = 0
    i_q_u = i_q_u1*1e6
    pspecs = hp.anafast(i_q_u, pol = True)
    t_e_b = hp.map2alm(i_q_u, lmax = ELLMAX, pol = True)
    first = msg.iterate_withbeam(Ndiag, i_q_u, B_pspec_cov,eta=0.825)
    sph2map = hp.alm2map((t_e_b[0], first[:NSPH], first[NSPH:]), NSIDE, lmax = ELLMAX, pol = True)
    second = hp.alm2cl(first[NSPH:], lmax = ELLMAX)
    list_of_b_spectra1.append(second)
    list_of_input_iqu.append(i_q_u)
    list_of_output_iqu.append(sph2map)
list_of_b_spectra = np.array(list_of_b_spectra1)

os.chdir('/users/PES0740/ucn3066/messenger/CMBS4/Sig_plus_N')
np.savez('Output SigplusNoise Spectra c_ell eta825_array{}.npz'.format(INDEX),list_of_b_spectra)
np.savez('Input SigplusNoise Maps{}.npz'.format(INDEX),list_of_input_iqu)
np.savez('Output SigplusNoise Maps{}.npz'.format(INDEX),list_of_output_iqu)
