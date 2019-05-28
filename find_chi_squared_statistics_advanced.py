import numpy as np
import healpy as hp
import os
import sys
os.chdir('/users/PES0740/ucn3066/messenger')
import messenger as msg

os.chdir('/users/PES0740/ucn3066/messenger/CMBS4/Noise')
Ncovmat = np.load('CMBS4 Ndiag.npz')['arr_0']
Ndiag = np.concatenate((Ncovmat[1,:],Ncovmat[2,:]),axis=0)

fwhm = 24.2
sigma_rad = (fwhm/(np.sqrt(8*np.log(2))))*(np.pi/(60*180))

os.chdir('/fs/project/PES0740/sky_yy/cmb/scalar')
talm = hp.read_alm('ffp10_unlensed_scl_cmb_000_tebplm_mc_0010.fits',hdu=1)
ealm = hp.read_alm('ffp10_unlensed_scl_cmb_000_tebplm_mc_0010.fits',hdu=2)
balm = hp.read_alm('ffp10_unlensed_scl_cmb_000_tebplm_mc_0010.fits',hdu=3)
i_q_u1 = hp.alm2map((talm,ealm,balm),nside=512,pol=True,sigma = sigma_rad)
i_q_u1[np.isnan(i_q_u1)] = 0
i_q_u = i_q_u1*10**6

dls = np.genfromtxt("/fs/project/PES0740/sky_yy/cmb/cls/ffp10_lensedCls.dat")
dls = np.insert(dls, [0], [1,0,0,0,0], axis = 0)
dls = np.insert(dls, [0], [0,0,0,0,0], axis = 0)
d_ell = np.array([dls[:,0]])
d_spectra = dls[:,1:]
Cls = np.array([d_spectra[i]*2*np.pi/(i * (i+1)) for i in range(len(d_spectra))])
c_ells = np.concatenate((np.transpose(d_ell), Cls), axis = 1)
c_ells_t = np.transpose(c_ells)
B_pspec_cov = c_ells_t[3,:1535]

s1, chi_squared1, lamlist1 = msg.iterate_untilchi_steps(Ndiag, i_q_u, B_pspec_cov, 1, 16, 3, min_chi = 1.75*10**6)
os.chdir('/users/PES0740/ucn3066/messenger/Chi_squared_stats')
np.savez('chi_square_maxlam16_minlam1_stepsize3_minchi1.75e6.npz',chi_squared1)
np.savez('lambda_vals_maxlam16_minlam1_stepsize3_minchi1.75e6.npz',lamlist1)

s2, chi_squared2, lamlist2 = msg.iterate_untilchi_eta(Ndiag, i_q_u, B_pspec_cov, 50, 0.6, min_chi = 1.75*10**6)
np.savez('chi_square_maxlam50_minchi1.75e6_eta06.npz',chi_squared2)
np.savez('lambda_vals_maxlam50_minchi1.75e6_eta06.npz',lamlist2)
