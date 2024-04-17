import os, sys
sys.path.append('/home/minghao/Research/Projects/JWST/qsohost_imaging/code/image_fitting/')

from startest import *

epsfdir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/epsf/'
startbl = Table.read(epsfdir+'/F200W_moda_starlist.vis.fits')

psfmc_startest_mock_step2(startbl, psfmcdir='/home/minghao/Research/Projects/JWST/qsohost_imaging/data/psfmc/starhosttest/J0148+0600/F200W_moda/', qname='J0148+0600', filt='F200W', F356Wvisits=[1,2,3,4], filtvisits=[1,2,3,4])
psfmc_startest_mock_step2(startbl, psfmcdir='/home/minghao/Research/Projects/JWST/qsohost_imaging/data/psfmc/starhosttest/J159-02/F200W_moda/', qname='J159-02', filt='F200W', F356Wvisits=[1,2], filtvisits=[1,2])
psfmc_startest_mock_step2(startbl, psfmcdir='/home/minghao/Research/Projects/JWST/qsohost_imaging/data/psfmc/starhosttest/J1120+0641/F200W_moda/', qname='J1120+0641', filt='F200W', F356Wvisits=[1,2,3], filtvisits=[1,2,3,4])

#for qname in ['J0148+0600', 'J159-02', 'J1120+0641']:
#    testdir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/psfmc/starhosttest/%s/%s_mod%s/'%(qname,filt,mod)

#    psfmc_startest_mock(startbl, testdir)

