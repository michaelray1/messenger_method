import numpy as np
import healpy as hp
import sys
import os
os.chdir('/users/PES0740/ucn3066/messenger/')
import messenger as msg

INDEX = int(sys.argv[1])

NSIDE = 512
ELLMAX = 3*NSIDE - 1
NPIX = 12*NSIDE**2
NSPH = int(np.sum(1.0 * np.arange(ELLMAX+1) + 1.0))

fwhm = 24.2
sigma_rad = (fwhm/(np.sqrt(8*np.log(2))))*(np.pi/(60*180))

Ncovmat = np.load('/users/PES0740/ucn3066/messenger/CMBS4/Noise/CMBS4 Ndiag.npz')['arr_0']
Ndiag = np.concatenate((Ncovmat[1,:],Ncovmat[2,:]),axis=0)
dls = np.genfromtxt("/fs/project/PES0740/sky_yy/cmb/cls/ffp10_lensedCls.dat")
dls = np.insert(dls, [0], [1,0,0,0,0], axis = 0)
dls = np.insert(dls, [0], [0,0,0,0,0], axis = 0)
d_ell = np.array([dls[:,0]])
d_spectra = dls[:,1:]  
Cls = np.array([d_spectra[i]*2*np.pi/(i * (i+1)) for i in range(len(d_spectra))])
c_ells = np.concatenate((np.transpose(d_ell), Cls), axis = 1)
c_ells_t = np.transpose(c_ells)

os.chdir('/users/PES0740/ucn3066/messenger/CMBS4/bpwf/BB2BB/')
bb_bpwf = np.load('take2_summed_bpwf.npz')['arr_0']

iqu_maps = np.empty([10,3,NPIX])
os.chdir('/fs/project/PES0740/sky_yy/cmb/scalar')
for i in np.arange(10*INDEX,10*INDEX+10):
    talm = hp.read_alm('ffp10_unlensed_scl_cmb_000_tebplm_mc_{:04d}.fits'.format(i),hdu=1)
    ealm = hp.read_alm('ffp10_unlensed_scl_cmb_000_tebplm_mc_{:04d}.fits'.format(i),hdu=2)
    balm = hp.read_alm('ffp10_unlensed_scl_cmb_000_tebplm_mc_{:04d}.fits'.format(i),hdu=3)
    iqu = hp.alm2map((talm,ealm,balm),nside=512,pol=True,sigma = sigma_rad)
    iqu_maps[i,:,:] = iqu

ells = np.arange(ELLMAX+1)
filt_ee_maps_bbout_dl = np.empty([10,ELLMAX+1])
binned_bbout = np.empty([10,10])
for i in range(10):
    filt_eemap = msg.iterate_withbeam(Ndiag, iqu_maps[i,:,:], c_ells_t[3,:1535], eta=8.25/10)
    filt_bbvals = filt_eemap[NSPH:]
    bb_outspec = hp.alm2cl(filt_bbvals, lmax = ELLMAX)
    filt_ee_maps_bbout_dl[i,:] = bb_outspec*ells*(ells+1)/(2*np.pi)
    binned_bbout[i,:] = msg.bin_pspec(bb_outspec,30,370,35)

sup_corrected_bbspec = binned_bbout/bb_bpwf

os.chdir('/users/PES0740/ucn3066/messenger/CMBS4/EE_only_sims')
np.savez('Out_supfac_corrected_binned_bb_pspec{}.npz'.format(INDEX),sup_corrected_bbspec)
np.savez('Out_raw_binned_bb_pspec{}.npz'.format(INDEX),binned_bbout)
np.savez('Out_bb_raw_pspec{}.npz'.format(INDEX),filt_ee_maps_bbout_dl)
np.savez('Input_iqu_maps{}.npz'.format(INDEX),iqu_maps)
