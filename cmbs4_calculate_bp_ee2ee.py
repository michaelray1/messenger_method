import healpy as hp
import numpy as np
import os
import sys
import messenger as msg

INDEX = int(sys.argv[1])

NSIDE = 512
NPIX = 12*NSIDE**2
ELLMAX = 3*NSIDE-1
NSPH = int(np.sum(1.0 * np.arange(ELLMAX+1) + 1.0))

FWHM = 24.2
sigma_rad = (FWHM/(np.sqrt(8*np.log(2))))*(np.pi/(60*180))
B_ell_squared = [np.e**((-1)*(index**2)*(sigma_rad**2)) for index in range(ELLMAX)]

dls = np.genfromtxt("/fs/project/PES0740/sky_yy/cmb/cls/ffp10_lensedCls.dat")
dls = np.insert(dls, [0], [1,0,0,0,0], axis = 0)
dls = np.insert(dls, [0], [0,0,0,0,0], axis = 0)
d_ell = np.array([dls[:,0]])
d_spectra = dls[:,1:]
Cls = np.array([d_spectra[i]*2*np.pi/(i * (i+1)) for i in range(len(d_spectra))])
c_ells = np.concatenate((np.transpose(d_ell), Cls), axis = 1)
c_ells_t = np.transpose(c_ells)
Ncovmat = np.load('/users/PES0740/ucn3066/messenger/CMBS4/Noise/CMBS4 Ndiag.npz')['arr_0']
Ndiag = np.concatenate((Ncovmat[1,:],Ncovmat[2,:]),axis=0)

os.chdir('/users/PES0740/ucn3066/messenger/CMBS4/bpwf/EE2EE')
tot_binned_espec = np.empty([6, 10])
i = 0
while i <= 5:
    input_pspec = np.zeros(ELLMAX)
    input_pspec[INDEX] = 1.0*B_ell_squared[INDEX]*2*np.pi/(INDEX*(INDEX+1))
    i_q_u = hp.synfast((np.zeros(ELLMAX),input_pspec,np.zeros(ELLMAX),np.zeros(ELLMAX)), NSIDE, pol=True, new=True)
    output_alms, bspecs, lam_list = msg.iterate_withbeam(Ndiag, i_q_u, c_ells_t[3,:1536], eta = 8.25/10)
    output_almsee = output_alms[:NSPH]
    output_espec = hp.alm2cl(output_almsee, lmax = ELLMAX)
    output_eedl = np.array([output_espec[j]*(j*(j+1))/(2*np.pi) for j in range(len(output_espec))])
    binned_espec_dl = msg.bin_pspec(output_eedl, 20, 370, 35)
    tot_binned_espec[i,:] = binned_espec_dl
    i += 1

np.savez('binned_ee2ee_bpwf_inputell{}.npz'.format(INDEX), tot_binned_espec)
