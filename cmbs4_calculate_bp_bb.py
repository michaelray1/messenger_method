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

os.chdir('/users/PES0740/ucn3066/messenger/CMBS4/bpwf/BB2BB')
tot_binned_bbdl = np.empty([370, 10])
list_of_output_bspec = np.empty([370,ELLMAX+1])
list_of_converted_dl_bspec = np.empty([370,ELLMAX+1])
for input_ell in range(370):
    print(input_ell)
    input_pspec = np.zeros(ELLMAX)
    input_pspec[input_ell] = 1.0*B_ell_squared[input_ell]*2*np.pi/(input_ell*(input_ell+1))
    i_q_u = hp.synfast((np.zeros(ELLMAX),np.zeros(ELLMAX),input_pspec,np.zeros(ELLMAX)), NSIDE, pol=True, new=True)
    output_almsbb = msg.iterate_withbeam(Ndiag, i_q_u, c_ells_t[3,:1535], eta=8.25/10)[NSPH:]
    output_bspec = hp.alm2cl(output_almsbb, lmax = ELLMAX)
    output_bbdl = np.array([output_bspec[i]*(i*(i+1))/(2*np.pi) for i in range(len(output_bspec))])
    binned_bspec_dl = msg.bin_pspec(output_bbdl, 20, 370, 35)
    tot_binned_bbdl[input_ell,:] = binned_bspec_dl
    list_of_output_bspec[input_ell,:] = output_bspec
    list_of_converted_dl_bspec[input_ell,:] = output_bbdl


np.savez('take2_binned_bb2bb_bpwf_24iter_sim{}.npz'.format(INDEX), tot_binned_bbdl)
np.savez('take2_list of output bspec{}.npz'.format(INDEX),list_of_output_bspec)
np.savez('take2_list of converted dl bspec{}.npz'.format(INDEX),list_of_converted_dl_bspec)
