import os, sys
sys.path.append('/home/minghao/Research/Projects/JWST/qsohost_imaging/code/image_fitting/')

from startest import *

filt = 'F115W'
mod = 'a'
size = 201
margin = 40


epsfdir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/epsf/'
testdir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/psfmc/startest/%s_mod%s/'%(filt,mod)
outputdir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/stars/%s_mod%s/'%(filt,mod)

os.system('mkdir -p %s'%testdir)
os.system('mkdir -p %s'%outputdir)

startbl = Table.read(epsfdir+'/%s_mod%s_starlist.vis.fits'%(filt, mod))
epsfstars = pickle.load(open(epsfdir+'/%s_mod%s_starlist.vis.dat'%(filt, mod), 'rb'))

#make_psf_for_startest(epsfstars, size, margin, outputdir)
#psfmc_startest(startbl, outputdir, testdir, size+1)
psfmc_startest_1p(startbl, outputdir, testdir, size+1)

# we now want to test the other bands
#psfmc_startest(startbl, outputdir, testdir, size+1,\
#               quasardir='/home/minghao/Research/Projects/JWST/qsohost_imaging/data/psfmc/J0148+0600/F356W/',\
#               visits=[1,2,3,4], subdir='fixed_J0148+0600')
#psfmc_startest(startbl, outputdir, testdir, size+1,\
#               quasardir='/home/minghao/Research/Projects/JWST/qsohost_imaging/data/psfmc/J1120+0641/F200W/',\
#               visits=[1,2,3,4], subdir='fixed_J1120+0641')
#psfmc_startest(startbl, outputdir, testdir, size+1,\
#               quasardir='/home/minghao/Research/Projects/JWST/qsohost_imaging/data/psfmc/J159-02/F356W/',\
#               visits=[1,2], subdir='fixed_J159-02')

