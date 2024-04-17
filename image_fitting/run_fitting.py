import os, sys
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table

from psfMC.ModelComponents import Configuration, Sky, PointSource, Sersic
from psfMC.distributions import Normal, Uniform, WeibullMinimum
from psfMC import model_galaxy_mcmc, load_database

def fileprep(datafile, psffile, psfmcdir):

    hdulist = fits.open(datafile)
    header = hdulist[1].header

    data = hdulist[1].data
    if data.shape[0]%2==1:
        data = data[1:,1:]
    hdu_img = fits.PrimaryHDU(data=data, header=header)
    hdu_img.writeto(psfmcdir+'/image0.fits', overwrite=True)

    origmask = hdulist[3].data
    if origmask.shape[0]%2==1:
        origmask = origmask[1:,1:]

    hdu_origmask = fits.PrimaryHDU(data=origmask, header=header)
    hdu_origmask.writeto(psfmcdir+'/origmask.fits', overwrite=True)
    if os.path.exists(psfmcdir+'/newmask.fits'):
        print('find a new mask')
        mask = fits.getdata(psfmcdir+'/newmask.fits')
    else:
        mask = origmask

    hdu_mask = fits.PrimaryHDU(data=mask, header=header)
    hdu_mask.writeto(psfmcdir+'/mask0.fits', overwrite=True)

    sigma = hdulist[2].data
    ivm = 1/sigma**2
    ivm[np.isnan(ivm)] = 1e13
    ivm[np.isinf(ivm)] = 1e13
    if ivm.shape[0]%2==1:
        ivm = ivm[1:,1:]

    hdu_ivm = fits.PrimaryHDU(data=ivm, header=header)
    hdu_ivm.writeto(psfmcdir+'/ivm0.fits', overwrite=True)

    hdulist_psf = fits.open(psffile)
    psfdata = hdulist_psf[0].data
    hdu_psf = fits.PrimaryHDU(data=psfdata)
    hdu_psf.writeto(psfmcdir+'/psf.fits', overwrite=True)

    ivmpsf = 1/hdulist_psf[1].data**2
    ivmpsf[np.isnan(ivmpsf)] = 1e13
    ivmpsf[np.isinf(ivmpsf)] = 1e13

    hdu_psfivm = fits.PrimaryHDU(data=ivmpsf)
    hdu_psfivm.writeto(psfmcdir+'/ivmpsf.fits', overwrite=True)


def generate_config_file(psfmcdir, hostinfolist, addbkg, addhost, addstring, total_mag=None):
    '''
    This function produces the model.py file that psfMC runs.
    '''
    print(psfmcdir)
    data = fits.getdata(psfmcdir+'/image0.fits')
    header = fits.getheader(psfmcdir+'/image0.fits')

    flux = np.sum(data)

    # get the zero point and pixel scale

    pixscale = header['CDELT1']*3600
    zeropoint = -6.10 - 2.5* np.log10(header['PIXAR_SR'])

    if total_mag is None:
        total_mag = -2.5*np.log10(flux) + zeropoint

    galmag_guess = total_mag + 4

    xmax, ymax = data.shape
    xc = xmax / 2
    yc = ymax / 2

    string_base = \
'''
import os, sys
import numpy as np

pixscale = %.3f

total_mag = %.2f
center = np.array((%d, %d))

max_shift_q = np.array((0.1, 0.1))/pixscale
max_shift_g = np.array((0.5, 0.5))/pixscale

zeropoint = %.4f
rmin = 0.05/pixscale
rmax = 1.0/pixscale

Configuration(obs_file='image0.fits',
          obsivm_file='ivm0.fits',
          psf_files='psf.fits',
          psfivm_files='ivmpsf.fits',
          mask_file='mask0.fits',
          mag_zeropoint=zeropoint)
'''%(pixscale, total_mag, xc, yc, zeropoint)

    # point source: always free
    string_ps = \
'''
PointSource(xy=Uniform(loc=center - max_shift_q, scale=2 * max_shift_q),
        mag=Uniform(loc=total_mag-1, scale=4))
'''

    # sky: always free
    string_sky = \
'''
# We can treat the sky as an unknown component if the subtraction is uncertain
Sky(adu=Uniform(loc=-0.05, scale=0.1))
'''

    # host: depends

    if hostinfolist is None:

        string_host = \
'''
# host component
Sersic(xy=Uniform(loc=center-max_shift_g, scale=2*max_shift_g),
       mag=Uniform(loc=total_mag+2, scale=4),
       reff=Uniform(loc=rmin, scale=rmax-rmin),
       reff_b=Uniform(loc=rmin, scale=rmax-rmin),
       index=1,
       angle=Uniform(loc=0, scale=180), angle_degrees=True)
'''
    else:
        print('fixed host')

        string_host = \
'''
Sersic(xy=(%.4f, %.4f),
       mag=Uniform(loc=20, scale=10),
       reff=%.4f,
       reff_b=%.4f,
       index=%.4f,
       angle=%.4f, angle_degrees=True)
'''%hostinfolist

    string = string_base + string_ps
    if addhost:
        string += string_host
    if addbkg:
        string += string_sky

    string += addstring

    modelfile = psfmcdir + '/model.py'
    print(modelfile)
    f = open(modelfile, 'w')
    f.write(string)

'''
def update_host_direct(best_dir, other_dir):
    result_best = generate_results_dict(best_dir)
    result_other = generate_results_dict(other_dir)

    header = fits.getheader(other_dir+'/image0.fits')
    pixscale = header['CDELT1']*3600

    # make a new list of params
    hostx_fix = (result_other['x_qso'] + result_best['x_gal'] - result_best['x_qso']) / pixscale
    hosty_fix = (result_other['y_qso'] + result_best['y_gal'] - result_best['y_qso']) / pixscale

    rmaj_fix = result_best['rmaj_gal'] / pixscale
    rmin_fix = result_best['rmin_gal'] / pixscale

    index_fix = result_best['index_gal']
    pa_fix = result_best['pa_gal']

    fixlist = (hostx_fix, hosty_fix, rmaj_fix, rmin_fix, index_fix, pa_fix)

    generate_config_file(datafile, psffile, psfmcdir,\
                         hostinfolist=hostinfolist, addbkg=True, addhost=True, addstring='')
'''

def generate_results_dict(direct, visits, fixed=False, psfonly=False):
    # read header to get the pixel scale
    directlist = [direct + '/visit%d'%v for v in visits]
    return generate_results_dirlist(directlist, fixed, psfonly)

def generate_results_dirlist(directlist, fixed, psfonly):
    header = fits.getheader(directlist[0]+'/image0.fits')

    pixscale = header['CDELT1']*3600
#    zeropoint = -6.10 - 2.5* np.log10(header['PIXAR_SR'])

    dblist = []
    for direct in directlist:
        db = load_database(direct+'/out_db.fits')
        select_idx = np.random.choice(range(len(db)), 10000)
        new_db = db[select_idx]
        dblist.append(new_db)

#        print(len(dblist[-1]))

    # extract the following quantities
    colnames = ['0_PointSource_xy', '0_PointSource_mag', '1_Sersic_xy',
        '1_Sersic_angle', '1_Sersic_reff', '1_Sersic_reff_b', '1_Sersic_mag', '1_Sersic_index']

    dblist = [load_database(direct+'/out_db.fits') for direct in directlist]

    x_qso_list = np.concatenate([db['0_PointSource_xy'][:,0] for db in dblist])
    y_qso_list = np.concatenate([db['0_PointSource_xy'][:,1] for db in dblist])

    # magnitude offset
    magoff = []
    for db in dblist:
        magoff.append(np.median(dblist[0]['0_PointSource_mag'])-np.median(db['0_PointSource_mag']))

    mag_qso_list = np.concatenate([dblist[index]['0_PointSource_mag'] + magoff[index] for index in range(len(dblist))])

#    print(psfonly)
    if psfonly:
        result_dict = {'x_qso': [np.median(x_qso_list)*pixscale, np.std(x_qso_list)*pixscale],\
                   'y_qso': [np.median(y_qso_list)*pixscale, np.std(y_qso_list)*pixscale],\
                   'mag_qso': [np.median(mag_qso_list), np.std(mag_qso_list)]}

    elif fixed:
        # build the dict
        mag_gal_list = np.concatenate([dblist[index]['1_Sersic_mag'] + magoff[index] for index in range(len(dblist))])

        result_dict = {'x_qso': [np.median(x_qso_list)*pixscale, np.std(x_qso_list)*pixscale],\
                   'y_qso': [np.median(y_qso_list)*pixscale, np.std(y_qso_list)*pixscale],\
                   'mag_qso': [np.median(mag_qso_list), np.std(mag_qso_list)],\
                   'mag_gal': [np.median(mag_gal_list), np.std(mag_gal_list)]}
    else:
        x_gal_list = np.concatenate([db['1_Sersic_xy'][:,0] for db in dblist])
        y_gal_list = np.concatenate([db['1_Sersic_xy'][:,1] for db in dblist])

        rmaj_list = np.concatenate([db['1_Sersic_reff'] for db in dblist])
        rmin_list = np.concatenate([db['1_Sersic_reff_b'] for db in dblist])
        pa_list = np.concatenate([db['1_Sersic_angle'] for db in dblist])

        ell_list = 1 - rmin_list / rmaj_list
        re_list = np.sqrt(rmin_list * rmaj_list)

        mag_gal_list = np.concatenate([dblist[index]['1_Sersic_mag'] + magoff[index] for index in range(len(dblist))])

        # build the dict
        result_dict = {'x_qso': [np.median(x_qso_list)*pixscale, np.std(x_qso_list)*pixscale],\
                   'y_qso': [np.median(y_qso_list)*pixscale, np.std(y_qso_list)*pixscale],\
                   'x_gal': [np.median(x_gal_list)*pixscale, np.std(x_gal_list)*pixscale],\
                   'y_gal': [np.median(y_gal_list)*pixscale, np.std(y_gal_list)*pixscale],\
                   'mag_qso': [np.median(mag_qso_list), np.std(mag_qso_list)],\
                   'mag_gal': [np.median(mag_gal_list), np.std(mag_gal_list)],\
                   'rmaj_gal': [np.median(rmaj_list)*pixscale, np.std(rmaj_list)*pixscale],\
                   'rmin_gal': [np.median(rmin_list)*pixscale, np.std(rmin_list)*pixscale],\
                   'pa_gal': [np.median(pa_list), np.std(pa_list)],\
                    'ell_gal':[np.median(ell_list), np.std(ell_list)],\
                    're_gal':[np.median(re_list), np.std(re_list)],\
                   'index_gal': [1, 0]}

#    print(result_dict)
    return result_dict

def run_mcmc(model_file, newfit=True, mc_args={'burn': 10000, 'iterations': 500, 'chains': 30}):
    output_name = model_file.replace('model', 'out').replace('.py', '')

    if newfit:
        os.system('rm %s_*'%output_name)

    cwd = os.getcwd()
    workdir = os.path.dirname(model_file)
    os.chdir(workdir)
    model_galaxy_mcmc(model_file, output_name=output_name, **mc_args)

def prepare_one_quasar_filt(qname, filt, datadir, epsfdir, outdir, visits,
                            hostinfolist=None, addbkg=True, addhost=True, addstring=''):

    if qname=='J1120+0641' and filt in ['F200W']:
        # J1120 has a bright star and the quasar fall on the spike of that star.
        # so we use "_sub.fits" images where the star has been removed.
        qsocubelist = [datadir+'/qso_filter%s_visit1_moda_sub.fits'%filt,\
                    datadir+'/qso_filter%s_visit2_moda_sub.fits'%filt,\
                    datadir+'/qso_filter%s_visit3_modb_sub.fits'%filt,\
                    datadir+'/qso_filter%s_visit4_modb_sub.fits'%filt]
    else:
        qsocubelist = [datadir+'/qso_filter%s_visit1_moda.fits'%filt,\
                    datadir+'/qso_filter%s_visit2_moda.fits'%filt,\
                    datadir+'/qso_filter%s_visit3_modb.fits'%filt,\
                    datadir+'/qso_filter%s_visit4_modb.fits'%filt]

    if qname=='J0100+2802':
        total_mag = 17
    elif qname == 'J1148+5251':
        total_mag = 19
    else:
        total_mag = None

    psflist = [epsfdir+'/epsf_%s_moda.fits'%(filt),\
            epsfdir+'/epsf_%s_moda.fits'%(filt),\
            epsfdir+'/epsf_%s_modb.fits'%(filt),\
            epsfdir+'/epsf_%s_modb.fits'%(filt)]

    for visit in visits:
        qsodir = outdir+'/visit%d'%visit

        os.system('mkdir -p %s'%qsodir)

        psffile = psflist[visit-1]
        qsofile = qsocubelist[visit-1]

        fileprep(qsofile, psffile, qsodir)
        generate_config_file(qsodir, hostinfolist=hostinfolist, addbkg=addbkg, addhost=addhost,\
                             addstring=addstring, total_mag=total_mag)

def do_one_quasar_step_one(qname, filters=['F356W', 'F200W', 'F115W'], visits=[1,2,3,4], **kwargs):

    # step 1: fit individual filter + visit, all free
    for filt in filters:
        datadir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/quasars/%s/'%(qname)
        psfmcdir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/psfmc/%s/%s/'%(qname,filt)
        epsfdir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/epsf/'

        string = addstring_for_quasars(qname, filt)
        prepare_one_quasar_filt(qname, filt, datadir, epsfdir, psfmcdir, visits=visits, addstring=string)

        for visit in visits:
            modelfile = psfmcdir+'/visit%d/model.py'%visit
            run_mcmc(modelfile, **kwargs)

def do_one_quasar_psf_only(qname, filters=['F356W', 'F200W', 'F115W'], visits=[1,2,3,4], **kwargs):

    # step 1: fit individual filter + visit, all free
    for filt in filters:
        datadir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/quasars/%s/'%(qname)
        psfmcdir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/psfmc/%s/%s_1p/'%(qname,filt)
        epsfdir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/epsf/'

#        string = addstring_for_quasars(qname, filt)
        prepare_one_quasar_filt(qname, filt, datadir, epsfdir, psfmcdir, visits=visits, addstring='', addhost=False)

        for visit in visits:
            modelfile = psfmcdir+'/visit%d/model.py'%visit
            run_mcmc(modelfile, **kwargs)

def do_one_quasar_step_two(qname, bestband='F356W', otherbands=['F200W', 'F115W'], visits=[1,2,3,4], bestvisits=[1,2,3,4], **kwargs):

    # step 2: pick up the best band, and fix the other parameters to the best band
    psfmcdir_best = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/psfmc/%s/%s/'%(qname,bestband)

    bestresult_dict = generate_results_dict(psfmcdir_best, bestvisits)

    for filt in otherbands:
        datadir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/quasars/%s/'%(qname)
        psfmcdir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/psfmc/%s/%s/'%(qname,filt)
        psfmcdir_fix = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/psfmc/%s/%s_fix2/'%(qname,filt)
        epsfdir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/epsf/'

        for visit in visits:
            # extract the info: xg, yg, reff, reff_b, index, angle
            pixscale = fits.getheader(psfmcdir+'/visit1/image0.fits', 0)['CDELT1']*3600
            filtresult_dict = generate_results_dict(psfmcdir, [visit])
            xg = ((bestresult_dict['x_gal'][0] - bestresult_dict['x_qso'][0]) + filtresult_dict['x_qso'][0])/pixscale
            yg = ((bestresult_dict['y_gal'][0] - bestresult_dict['y_qso'][0]) + filtresult_dict['y_qso'][0])/pixscale

            reff = bestresult_dict['rmaj_gal'][0] / pixscale
            reff_b = bestresult_dict['rmin_gal'][0] / pixscale

            angle = bestresult_dict['pa_gal'][0]
            index = bestresult_dict['index_gal'][0]

            string = addstring_for_quasars(qname, filt)

            hostinfolist = (xg, yg, reff, reff_b, index, angle)
            prepare_one_quasar_filt(qname, filt, datadir, epsfdir, psfmcdir_fix, visits=[visit],\
                                    hostinfolist=hostinfolist, addstring=string)

            modelfile = psfmcdir_fix+'/visit%d/model.py'%visit
            run_mcmc(modelfile, **kwargs)


def addstring_for_quasars(qname, filt):

    # J0148: add the two companions

    if qname == 'J0148+0600':
        if filt == 'F356W':
            string = \
'''
# first companion
max_shift_comp = 5
center1 = np.array((11.21,87.95))
Sersic(xy=Uniform(loc=center1-max_shift_comp, scale=2*max_shift_comp),
       mag=Uniform(loc=22, scale=3),
       reff=Uniform(loc=1, scale=30),
       reff_b=Uniform(loc=1, scale=30),
       index=Uniform(loc=0.5, scale=5),
       angle=Uniform(loc=0, scale=180), angle_degrees=True)

# second companion
center2 = np.array((80.55,29.8))
Sersic(xy=Uniform(loc=center2-max_shift_comp, scale=2*max_shift_comp),
       mag=Uniform(loc=22, scale=3),
       reff=Uniform(loc=1, scale=30),
       reff_b=Uniform(loc=1, scale=30),
       index=Uniform(loc=0.5, scale=5),
       angle=Uniform(loc=0, scale=180), angle_degrees=True)
'''
        else:
            string = \
'''
# first companion
max_shift_comp = 10
center1 = np.array((11.21,87.95))*2
Sersic(xy=Uniform(loc=center1-max_shift_comp, scale=2*max_shift_comp),
       mag=Uniform(loc=22, scale=3),
       reff=Uniform(loc=1, scale=60),
       reff_b=Uniform(loc=1, scale=60),
       index=Uniform(loc=0.5, scale=5),
       angle=Uniform(loc=0, scale=180), angle_degrees=True)

# second companion
center2 = np.array((80.55,29.8))*2
Sersic(xy=Uniform(loc=center2-max_shift_comp, scale=2*max_shift_comp),
       mag=Uniform(loc=22, scale=3),
       reff=Uniform(loc=1, scale=60),
       reff_b=Uniform(loc=1, scale=60),
       index=Uniform(loc=0.5, scale=5),
       angle=Uniform(loc=0, scale=180), angle_degrees=True)
'''

    elif qname == 'J159-02':
        if filt == 'F356W':
            string = \
'''
# first companion
max_shift_comp = 5
center1 = np.array((87,17))
Sersic(xy=Uniform(loc=center1-max_shift_comp, scale=2*max_shift_comp),
       mag=Uniform(loc=25, scale=3),
       reff=Uniform(loc=1, scale=30),
       reff_b=Uniform(loc=1, scale=30),
       index=Uniform(loc=0.5, scale=8),
       angle=Uniform(loc=0, scale=180), angle_degrees=True)
'''
        else:
            string = \
'''
# first companion
max_shift_comp = 10
center1 = np.array((87, 17))*2
Sersic(xy=Uniform(loc=center1-max_shift_comp, scale=2*max_shift_comp),
       mag=Uniform(loc=25, scale=3),
       reff=Uniform(loc=1, scale=60),
       reff_b=Uniform(loc=1, scale=60),
       index=Uniform(loc=0.5, scale=8),
       angle=Uniform(loc=0, scale=180), angle_degrees=True)
'''

    else:
        string = ''

    return string


if __name__=='__main__':
    main()

