from build_epsf import *
from dataprep import *
from run_fitting import *
from copy import deepcopy

def make_psf_for_startest(epsfstars, size, margin, output_dir):

    for index in range(len(epsfstars)):
        subepsfstars = deepcopy(epsfstars)
        subepsfstars.__delitem__(index)
        output = output_dir + '/psf%d.fits'%index

        epsf, fitted_stars = build_epsf_onestep(subepsfstars, size, margin, None)
        epsf_error = generate_epsf_error(fitted_stars, epsf)

        epsf_data = epsf.data[margin:-margin,margin:-margin]
        epsf_data[epsf_data<0] = 0
        epsf_hdu = fits.PrimaryHDU(data=epsf_data)
        epsferr_hdu = fits.ImageHDU(data=epsf_error[margin:-margin,margin:-margin])

        epsf_hdulist = fits.HDUList([epsf_hdu, epsferr_hdu])
        epsf_hdulist.writeto(output, overwrite=True)


def psfmc_startest(startbl, cutoutdir, psfmcdir, size,\
                  quasardir=None, visits=None, subdir=None):

    if not os.path.exists(cutoutdir):
        os.system('mkdir -p %s'%cutoutdir)
    if not os.path.exists(psfmcdir):
        os.system('mkdir -p %s'%psfmcdir)

    for index in range(len(startbl)):
        filename = startbl['ImageName'][index]
        starx = startbl['x'][index]
        stary = startbl['y'][index]

        pos = [starx, stary]
        starfile = cutoutdir + '/star%d.fits'%index

#        ncimg = NIRCamImage(imagename=filename)
#        cube = ncimg.make_galfit_cube(position=pos, size=size, output=starfile)

        psffile = cutoutdir + '/psf%d.fits'%index
        psfmcdir_thisstar = psfmcdir + '/star%d/'%index
        psfmcdir_thisstar_fix = psfmcdir + '/%s/star%d'%(subdir,index)

        if quasardir is None:
            os.system('mkdir -p %s'%psfmcdir_thisstar)
#            fileprep(starfile, psffile, psfmcdir_thisstar)

            generate_config_file(psfmcdir_thisstar, addbkg=True, addhost=True,\
                            hostinfolist=None, addstring='')
            modelfile = psfmcdir_thisstar+'/model.py'

        else:
            os.system('mkdir -p %s'%psfmcdir_thisstar_fix)
            fileprep(starfile, psffile, psfmcdir_thisstar_fix)

            bestresult_dict = generate_results_dict(quasardir, visits)

            pixscale = fits.getheader(psfmcdir_thisstar+'/image0.fits', 0)['CDELT1']*3600

            filtresult_dict = generate_results_dirlist([psfmcdir_thisstar])

            xg = ((bestresult_dict['x_gal'][0] - bestresult_dict['x_qso'][0]) + filtresult_dict['x_qso'][0])/pixscale
            yg = ((bestresult_dict['y_gal'][0] - bestresult_dict['y_qso'][0]) + filtresult_dict['y_qso'][0])/pixscale

            reff = bestresult_dict['rmaj_gal'][0] / pixscale
            reff_b = bestresult_dict['rmin_gal'][0] / pixscale

            angle = bestresult_dict['pa_gal'][0]
            index = bestresult_dict['index_gal'][0]

            hostinfolist = (xg, yg, reff, reff_b, index, angle)

            generate_config_file(psfmcdir_thisstar_fix, addbkg=True, addhost=True,\
                            hostinfolist=hostinfolist, addstring='')

            modelfile = psfmcdir_thisstar_fix+'/model.py'
        run_mcmc(modelfile)


def psfmc_startest_1p(startbl, cutoutdir, psfmcdir, size):

    if not os.path.exists(cutoutdir):
        os.system('mkdir -p %s'%cutoutdir)
    if not os.path.exists(psfmcdir):
        os.system('mkdir -p %s'%psfmcdir)

    for index in range(len(startbl)):
        direct_1p1s = psfmcdir + '/star%d/'%index
        direct_1p = psfmcdir + '/star%d_1p/'%index

        if os.path.exists(direct_1p1s):
            os.system('cp -r %s %s'%(direct_1p1s, direct_1p))

        else:
            filename = startbl['ImageName'][index]
            starx = startbl['x'][index]
            stary = startbl['y'][index]

            pos = [starx, stary]
            starfile = cutoutdir + '/star%d.fits'%index

            ncimg = NIRCamImage(imagename=filename)
            cube = ncimg.make_galfit_cube(position=pos, size=size, output=starfile)

            psffile = cutoutdir + '/psf%d.fits'%index
            psfmcdir_thisstar = psfmcdir + '/star%d_1p/'%index

            os.system('mkdir -p %s'%psfmcdir_thisstar)
            fileprep(starfile, psffile, psfmcdir_thisstar)

        generate_config_file(direct_1p, addbkg=True, addhost=False,\
                        hostinfolist=None, addstring='')
        modelfile = direct_1p+'/model.py'
        run_mcmc(modelfile, mc_args={'burn': 1000, 'iterations': 500, 'chains': 10})


def psfmc_startest_mock(startbl, psfmcdir, psfonly=False):

    '''
    This function runs the image fitting algorithm for simulated quasar images.
    The input parameter psfmcdir should contain subdirectories named "star%d" that are produced by the "simulate_host.py" file.
    '''

    for index in range(len(startbl)):
        psfmcdir_thisstar = psfmcdir + '/star%d/'%index
        psfmcdir_thisstar_1p = psfmcdir + '/star%d_1p/'%index

        if not psfonly:
            modelfile = psfmcdir_thisstar+'/model.py'
            generate_config_file(psfmcdir_thisstar, addbkg=True, addhost=True,\
                        hostinfolist=None, addstring='')

            print('Running 1p+1s for %s'%modelfile)
            run_mcmc(modelfile, mc_args={'burn': 10000, 'iterations': 500, 'chains': 40})

        else:
            modelfile_1p = psfmcdir_thisstar_1p+'/model.py'
            generate_config_file(psfmcdir_thisstar_1p, addbkg=True, addhost=False,\
                            hostinfolist=None, addstring='')

            print('Running 1p for %s'%modelfile_1p)
            run_mcmc(modelfile_1p, mc_args={'burn': 1000, 'iterations': 500, 'chains': 10})

def psfmc_startest_mock_step2(startbl, psfmcdir, qname, filt, F356Wvisits, filtvisits, psfonly):

    '''
    This function runs the image fitting algorithm for simulated quasar images.
    The input parameter psfmcdir should contain subdirectories named "star%d" that are produced by the "simulate_host.py" file.
    '''

    # step 2: pick up the best band, and fix the other parameters to the best band
    F356Wdir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/psfmc/%s/F356W/'%(qname)
    filtdir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/psfmc/%s/%s_fix/'%(qname, filt)

    bestfitinfo_F356W = generate_results_dict(F356Wdir, F356Wvisits)
    bestfitinfo_filt = generate_results_dict(filtdir, filtvisits, fixed=True)

    pixscale = fits.getheader(filtdir+'/visit1/image0.fits', 0)['CDELT1']*3600
    xg = ((bestfitinfo_F356W['x_gal'][0] - bestfitinfo_F356W['x_qso'][0]) + bestfitinfo_filt['x_qso'][0])/pixscale
    yg = ((bestfitinfo_F356W['y_gal'][0] - bestfitinfo_F356W['y_qso'][0]) + bestfitinfo_filt['y_qso'][0])/pixscale

    reff = bestfitinfo_F356W['rmaj_gal'][0] / pixscale
    reff_b = bestfitinfo_F356W['rmin_gal'][0] / pixscale

    angle = bestfitinfo_F356W['pa_gal'][0]
    index = bestfitinfo_F356W['index_gal'][0]

    # generate the mock
    hostinfolist = (xg, yg, reff, reff_b, index, angle)

    for index in range(len(startbl)):

        if psfonly:
            psfmcdir_thisstar = psfmcdir + '/star%d_1p/'%index
            addhost = False
        else:
            psfmcdir_thisstar = psfmcdir + '/star%d/'%index
            addhost = True

        modelfile = psfmcdir_thisstar+'/model.py'
        generate_config_file(psfmcdir_thisstar, addbkg=True, addhost=addhost,\
                        hostinfolist=hostinfolist, addstring='')

        if psfonly:
            run_mcmc(modelfile, mc_args={'burn': 1000, 'iterations': 500, 'chains': 10})
        else:
            run_mcmc(modelfile, mc_args={'burn': 2000, 'iterations': 500, 'chains': 20})

#        break


def main():
    filt = 'F115W'
    mod='a'
    size = 201
    margin = 40

    run_star_test(filt, mod, size, margin)


if __name__=='__main__':
    main()
