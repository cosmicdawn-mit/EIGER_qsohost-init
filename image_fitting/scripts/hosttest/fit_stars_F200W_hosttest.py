import os, sys
sys.path.append('/home/minghao/Research/Projects/JWST/qsohost_imaging/code/image_fitting/')

from startest import *

epsfdir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/epsf/'
filt = 'F200W'

for mod in 'ab':

    startbl = Table.read(epsfdir+'/%s_mod%s_starlist.vis.fits'%(filt, mod))

    qname = 'J0148+0600'
    testdir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/psfmc/starhosttest_v2/%s/%s_mod%s/'%(qname,filt,mod)
    psfmc_startest_mock_step2(startbl, psfmcdir=testdir, qname=qname, filt=filt,\
                              F356Wvisits=[1,2,3,4], filtvisits=[1,2,3,4], psfonly=True)

    qname = 'J159-02'
    testdir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/psfmc/starhosttest_v2/%s/%s_mod%s/'%(qname,filt,mod)
    psfmc_startest_mock_step2(startbl, psfmcdir=testdir, qname=qname, filt=filt,\
                              F356Wvisits=[1,2], filtvisits=[1,2], psfonly=True)

    qname = 'J1120+0641'
    testdir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/psfmc/starhosttest_v2/%s/%s_mod%s/'%(qname,filt,mod)
    psfmc_startest_mock_step2(startbl, psfmcdir=testdir, qname=qname, filt=filt,\
                              F356Wvisits=[1,2,3], filtvisits=[1,2,3,4], psfonly=True)

