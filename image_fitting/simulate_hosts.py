import os, sys
import numpy as np
from astropy.io import fits
from astropy.table import Table

import matplotlib.pyplot as plt
from psfMC import model_galaxy_mcmc, load_database
# procedure: use stars + the best fit host parameters to produce a simulated image

from run_fitting import *

def simulate_image_psfMC(star_dir, source_info_list):
    header = fits.getheader(star_dir+'/image0.fits')
    zeropoint = -6.10 - 2.5* np.log10(header['PIXAR_SR'])

    # read general info
    filestr = \
'''
Configuration(obs_file='image0.fits',
          obsivm_file='ivm0.fits',
          psf_files='psf.fits',
          psfivm_files='ivmpsf.fits',
          mask_file='mask0.fits',
          mag_zeropoint=%.4f)
'''%(zeropoint)

    for source_info in source_info_list:
        if source_info['type']=='PSF':
            string = \
'''
PointSource(xy=(%.4f, %.4f), mag=%.4f)
'''%(source_info['x'], source_info['y'], source_info['mag'])

        elif source_info['type']=='Sersic':
            string = \
'''
Sersic(xy=(%.4f, %.4f), mag=%.4f, reff=%.4f, reff_b=%.4f, index=%.4f, angle=%.4f, angle_degrees=True)
'''%(source_info['x'], source_info['y'], source_info['mag'], source_info['reff'], source_info['reff_b'], source_info['index'], source_info['angle'])

    filestr = filestr + string
    skystr = 'Sky(adu=Uniform(loc=-0.005, scale=0.01))\n'
    filestr = filestr + skystr

    # write to a file
    f = open(star_dir+'/model.py', 'w')
    f.write(filestr)
    f.close()

    # run psfMC
    mc_args={'burn': 1, 'iterations': 1, 'chains': 2}
    modelfile = 'model.py'
    run_mcmc(star_dir+'/'+modelfile, mc_args=mc_args)

    # the resulting file should be out_convolved_model.fits
    modeled_galaxy = fits.getdata(star_dir+'/out_convolved_model.fits')
    starimg = fits.getdata(star_dir+'/image0.fits')
    starmag = -2.5*np.log10(np.sum(starimg)) + zeropoint
    scalefactor = 10**((starmag - source_info_list[0]['mag'])/2.5)

    # test the scale factor
    newmag = -2.5*np.log10(np.sum(starimg*scalefactor)) + zeropoint
    print('newmag: %.4f, oldmag: %.4f'%(newmag, source_info_list[0]['mag']))

    hostmag = -2.5*np.log10(np.sum(modeled_galaxy)) + zeropoint
    print('hostmag: %.4f, expected: %.4f'%(hostmag, source_info['mag']))

    allimg = modeled_galaxy + starimg*scalefactor
    mockheader = header
    mockheader['SCALE'] = scalefactor
    fits.writeto(star_dir+'/mock0.fits', allimg, mockheader, overwrite=True)

    orig_ivm_hdu = fits.open(star_dir+'/ivm0.fits')
    orig_ivm_hdu[0].data = orig_ivm_hdu[0].data / scalefactor**2

    os.system('mv %s/image0.fits %s/image0_old.fits'%(star_dir, star_dir))
    os.system('mv %s/ivm0.fits %s/ivm0_old.fits'%(star_dir, star_dir))

    orig_ivm_hdu.writeto(star_dir+'/ivm0.fits', overwrite=True)
    
    os.system('mv %s/mock0.fits %s/image0.fits'%(star_dir, star_dir))

    return 0


def psfmc_startest_mock(startbl, qsoname, filt, psfmcdir):
    if filt=='F356W':
        datadir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/psfmc/%s/%s/'%(qsoname,filt)
    else:

        F356Wdir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/psfmc/%s/F356W/'%(qsoname)
        datadir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/psfmc/%s/%s_fix/'%(qsoname,filt)

    if qsoname == 'J159-02':
        visits = [1,2]
    elif qsoname == 'J1120+0641' and filt=='F356W':
        visits = [1,2,3]
    else:
        visits = [1,2,3,4]

    if filt == 'F356W':
        bestfitinfo = generate_results_dict(datadir, visits)
    else:
        raise NotImplementedError('Estimating F200W and F115W are not in the plan now.')

    #convert it to the form needed

    if filt=='F356W':
        pixscale = 0.03
    else:
        pixscale = 0.015

    psfinfo = {'type':'PSF', 'x':bestfitinfo['x_qso'][0]/pixscale, 'y':bestfitinfo['y_qso'][0]/pixscale,\
               'mag':bestfitinfo['mag_qso'][0]}
    hostinfo = {'type':'Sersic', 'x':bestfitinfo['x_gal'][0]/pixscale, 'y':bestfitinfo['y_gal'][0]/pixscale,\
                'mag':bestfitinfo['mag_gal'][0], \
                'reff':bestfitinfo['rmaj_gal'][0]/pixscale, 'reff_b':bestfitinfo['rmin_gal'][0]/pixscale, \
                'index':bestfitinfo['index_gal'][0], 'angle':bestfitinfo['pa_gal'][0]}

    source_info_list = [psfinfo, hostinfo]

    print(source_info_list)

    for index in range(len(startbl)):

        psfmcdir_thisstar = psfmcdir + '/star%d/'%index
        # simulate the image
#        simulate_image_psfMC(psfmcdir_thisstar, source_info_list)

        os.system('rm -r %s/star%d_1p/'%(psfmcdir, index))
        os.system('cp -r %s %s/star%d_1p/'%(psfmcdir_thisstar, psfmcdir, index))


def psfmc_startest_mockSW(startbl, qsoname, filt, psfmcdir):
    # first, read the best-fit results from F356W

    F356Wdir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/psfmc/%s/F356W/'%(qsoname)
    datadir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/psfmc/%s/%s_fix/'%(qsoname,filt)

    if qsoname == 'J159-02':
        F356Wvisits = [1, 2]
        visits = [1, 2]
    elif qsoname == 'J1120+0641':
        F356Wvisits = [1,2,3]
        visits = [1,2,3,4]
    else:
        visits = [1,2,3,4]
        F356Wvisits = [1, 2, 3, 4]

    bestfitinfo_F356W = generate_results_dict(F356Wdir, F356Wvisits)
    bestfitinfo_filt = generate_results_dict(datadir, visits, fixed=True)

    pixscale = fits.getheader(datadir+'/visit1/image0.fits', 0)['CDELT1']*3600
    xg = ((bestfitinfo_F356W['x_gal'][0] - bestfitinfo_F356W['x_qso'][0]) + bestfitinfo_filt['x_qso'][0])/pixscale
    yg = ((bestfitinfo_F356W['y_gal'][0] - bestfitinfo_F356W['y_qso'][0]) + bestfitinfo_filt['y_qso'][0])/pixscale

    reff = bestfitinfo_F356W['rmaj_gal'][0] / pixscale
    reff_b = bestfitinfo_F356W['rmin_gal'][0] / pixscale

    angle = bestfitinfo_F356W['pa_gal'][0]
    index = bestfitinfo_F356W['index_gal'][0]

    # generate the mock
    psfinfo = {'type':'PSF', 'x':bestfitinfo_filt['x_qso'][0]/pixscale, 'y':bestfitinfo_filt['y_qso'][0]/pixscale,\
               'mag':bestfitinfo_filt['mag_qso'][0]}
    hostinfo = {'type':'Sersic', 'x':xg, 'y':yg, 'mag':bestfitinfo_filt['mag_gal'][0],\
                'reff':reff, 'reff_b':reff_b, 'index':1, 'angle':angle}

    source_info_list = [psfinfo, hostinfo]

    print(source_info_list)

    for index in range(len(startbl)):

        psfmcdir_thisstar = psfmcdir + '/star%d/'%index
        print(psfmcdir_thisstar)
        # simulate the image
#        simulate_image_psfMC(psfmcdir_thisstar, source_info_list)

        os.system('rm -r %s/star%d_1p/'%(psfmcdir, index))
        os.system('cp -r %s %s/star%d_1p/'%(psfmcdir_thisstar, psfmcdir, index))

def simulate_one_quasar(qname,filt,mod, SW=False):

    epsfdir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/epsf/'
    testdir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/psfmc/startest/%s_mod%s/'%(filt,mod)
    mockdir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/psfmc/starhosttest_v2/%s/'%(qname)

    print(testdir, mockdir)
    os.system('mkdir -p %s'%mockdir)

#    os.system('rm -r %s/%s_mod%s'%(mockdir,filt,mod))
#    os.system('cp -r %s/ %s'%(testdir, mockdir))

    startbl = Table.read(epsfdir+'/%s_mod%s_starlist.vis.fits'%(filt, mod))

    if filt in ['F200W', 'F115W']:
        psfmc_startest_mockSW(startbl, qname, filt, psfmcdir=mockdir+'%s_mod%s/'%(filt,mod))
    elif filt in ['F356W']:
        psfmc_startest_mock(startbl, qname, filt, psfmcdir=mockdir+'%s_mod%s/'%(filt,mod))


def main():
    qname_list = ['J0148+0600', 'J159-02', 'J1120+0641']
    mod_list = ['a', 'b']

    filt_list = ['F356W', 'F200W', 'F115W']
#    filt_list = ['F200W']

    for qname in qname_list:
        for filt in filt_list:
            for mod in mod_list:
                simulate_one_quasar(qname,filt,mod)

if __name__=='__main__':
    main()
