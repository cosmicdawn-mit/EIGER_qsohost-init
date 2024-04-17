import os, sys
sys.path.append('/home/minghao/Research/Projects/JWST/qsohost_imaging/code/image_fitting/')

from startest import *

filt = 'F356W'
mod = 'a'
size = 101
margin = 20

epsfdir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/epsf/'
testdir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/psfmc/startest/%s_mod%s/'%(filt,mod)
outputdir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/stars/%s_mod%s/'%(filt,mod)

os.system('mkdir -p %s'%testdir)
os.system('mkdir -p %s'%outputdir)

startbl = Table.read(epsfdir+'/%s_mod%s_starlist.vis.fits'%(filt, mod))
epsfstars = pickle.load(open(epsfdir+'/%s_mod%s_starlist.vis.dat'%(filt, mod), 'rb'))

#make_psf_for_startest(epsfstars, size, margin, outputdir)
psfmc_startest(startbl[:3], outputdir, testdir, size+1)
#psfmc_startest_1p(startbl, outputdir, testdir, size+1)

