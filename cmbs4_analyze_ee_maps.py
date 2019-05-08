import numpy as np
import healpy as hp
import messenger as msg
import sys
import os

NSIDE = 512
ELLMAX = 3*NSIDE - 1
NPIX = 12*NSIDE**2

sigma = 24.2
sigma_rad = 

ee_maps = np.empty([100,NPIX])
os.chdir('/fs/project/PES0740/sky_yy/cmb/scalar')
for i in range(100):
    alm = hp.read_alm('ffp10_unlensed_scl_cmb_000_tebplm_mc_{:04d}.fits'.format(i))
    ee_map = hp.alm2map(alm,nside=512,sigma = sigma_rad)
    ee_maps[i,:] = ee_map

filt_ee_maps = np.empty([100,NPIX])
for i in range(100):
    filt_eemap = msg.iterate_withbeam(ee_maps[i,:]
