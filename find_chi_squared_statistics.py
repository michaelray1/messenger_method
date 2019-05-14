import numpy as np
import healpy as hp
import os
import sys
os.chdir('/users/PES0740/ucn3066/messenger')
import messenger as msg

os.chdir('/users/PES0740/ucn3066/messenger/CMBS4/Noise')
Ncovmat = np.load('CMBS4 Ndiag.npz')['arr_0']
Ndiag = np.concatenate((Ncovmat[1,:],Ncovmat[2,:]),axis=0)

os.chdir('/users/PES0740/ucn3066/CMBS4_maps')
i_q_u1 = hp.read_map('/users/PES0740/ucn3066/CMBS4_maps/cmbs4_04p00_comb_f095_b24_ellmin30_map_0512_mc_0000.fits',field=(0,1,2))
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

s1, chi_squared1 = msg.iterate_withbeam(Ndiag, i_q_u, B_pspec_cov, eta = 0.825)
s2, chi_squared2 = msg.iterate_fixedlam(Ndiag, i_q_u, B_pspec_cov, lambdaone = 35)
s3, chi_squared3 = msg.iterate_fixedlam(Ndiag, i_q_u, B_pspec_cov, lambdaone = 20)
s4, chi_squared4 = msg.iterate_fixedlam(Ndiag, i_q_u, B_pspec_cov, lambdaone = 5)
s5, chi_squared5, lamlist = msg.iterate_declam(Ndiag, i_q_u, B_pspec_cov, 1, 22, 3)

os.chdir('/users/PES0740/ucn3066/messenger/Chi_squared_stats')
np.savez('chi_square_eta825.npz',chi_squared1)
np.savez('chi_square_lam35.npz',chi_squared2)
np.savez('chi_square_lam20.npz',chi_squared3)
np.savez('chi_square_lam5.npz',chi_squared4)
np.savez('chi_square_maxl22_minl1_step3.npz',chi_squared5)
np.savez('lambda_vals.npz',lamlist)
