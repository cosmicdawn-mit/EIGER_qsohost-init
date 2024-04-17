import os, sys
sys.path.append('/home/minghao/Research/Projects/JWST/qsohost_imaging/code/image_fitting/')

from startest import *


epsfdir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/epsf/'
filt = 'F356W'

for mod in 'ab':

    startbl = Table.read(epsfdir+'/%s_mod%s_starlist.vis.fits'%(filt, mod))

    for qname in ['J0148+0600', 'J1120+0641', 'J159-02']:
        testdir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/psfmc/starhosttest_v2/%s/%s_mod%s/'%(qname,filt,mod)
        psfmc_startest_mock(startbl, testdir, psfonly=True)

