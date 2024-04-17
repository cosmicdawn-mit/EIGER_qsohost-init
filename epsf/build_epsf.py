import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pickle

from astropy.io import fits
from astropy.visualization import simple_norm
from astropy.nddata import NDData
from astropy.table import Table, vstack
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.coordinates import SkyCoord

from photutils.detection import DAOStarFinder
from photutils.psf import extract_stars
from photutils.psf import EPSFBuilder, EPSFModel, EPSFStars, EPSFStar
from photutils.centroids import centroid_1dg, centroid_2dg, centroid_com
from photutils.aperture import aperture_photometry, CircularAperture

sys.path.append(os.path.abspath('../'))
from dataprep import cutout_image

from galight.tools.measure_tools import measure_FWHM

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

'''
Utility functions
'''
def measure_center_gauss(imagename, xc, yc, size=21):
    cutout_img = cutout_image(imagename, [xc, yc], size=21, n_ext=1)
    cutout_nse = cutout_image(imagename, [xc, yc], size=21, n_ext=2)

    mask = (cutout_img.data==0)

    center_new = centroid_2dg(data=cutout_img.data, mask=mask)
    x0, y0 = cutout_img.origin_original

    xcnew = x0 + center_new[0]
    ycnew = y0 + center_new[1]

    return xcnew, ycnew

'''
Functions to select PSF stars
'''
def preselect_psf(imagename, sources, peaklim, fwhm_range, edge=500, radius=51):
    image = fits.getdata(imagename, 1)

    selected_index = []
    ymax, xmax = image.shape

    for index in range(len(sources)):
        xc = sources['xcentroid'][index]
        yc = sources['ycentroid'][index]
        peak = sources['peak'][index]
        flux = sources['flux'][index]

        # not close to the edge
        if xc<edge or yc<edge or xc>xmax-edge or yc>ymax-edge:
            continue

        # bright enough
        if peak<peaklim[0] or peak>peaklim[1]:
            continue

        # FWHM in a certain range
        cutout_hdu = cutout_image(imagename, position=[xc, yc], size=51)
        try:
            FWHMlist = measure_FWHM(cutout_hdu.data)
        except:
            print('bad FWHM measurement')
            print(cutout_hdu.data)
            plt.imshow(cutout_hdu.data)
            plt.show()
            continue

        FWHM_avg = np.mean(FWHMlist)

        if FWHM_avg<fwhm_range[0] or FWHM_avg>fwhm_range[1]:
            continue

        selected_index.append(index)

    return selected_index

def visual_inspection(imagename, sources, preselect_index, cutout_size=101, quasar_position=None, epsf_init=None, maglim=[18,21], magzp=29):
    finalselect_x = []
    finalselect_y = []

    fluxlist = []
    localbacklist = []

    # make EPSFStars for plotting
    nddata = NDData(fits.getdata(imagename, 1))

    for index in preselect_index:
        xc = sources['xcentroid'][index]
        yc = sources['ycentroid'][index]

        cutout = cutout_image(imagename, [xc, yc], size=cutout_size)
        x0, y0 = cutout.origin_original
        xcnew, ycnew = measure_center_gauss(imagename, xc, yc)

        startbl = Table({'x': [xcnew], 'y':[ycnew]})
        star = extract_stars(nddata, startbl, size=cutout_size)

        xccut = xcnew - x0
        yccut = ycnew - y0

        if (xccut - int(cutout_size/2))**2 + (yccut - int(cutout_size/2))**2 > 2**2:
            continue

        if quasar_position is not None:
            xq, yq = quasar_position
            dmin = np.sqrt(np.nanmin((xcnew - xq)**2 + (ycnew - yq)**2))
            if np.nanmin((xcnew - xq)**2 + (ycnew - yq)**2) < 50**2:
                continue

        try:
            meanFWHM = np.mean(measure_FWHM(cutout.data))
        except:
            continue

        aperture = CircularAperture([xccut, yccut], r=20.0) 
        flux = aperture_photometry(cutout.data, aperture)['aperture_sum'][0]

        mag = -2.5*np.log10(flux) + magzp

        # reject faint ones
        if mag < maglim[0] or mag > maglim[1]:
            continue

        if epsf_init is None:
            fig, ax = plt.subplots()
            norm = simple_norm(cutout.data, 'log', percent=99.5)
            ax.imshow(cutout.data, origin='lower', norm=norm)
            ax.plot(xcnew-x0, ycnew-y0, 'k+')
            ax.axis('off')
            ax.set_title('xc=%.2f, yc=%.2f\nmag=%.2f, FWHM=%.2f'%(xc,yc,mag,meanFWHM))
            plt.tight_layout()
            plt.show()

            localback = 0

        else:
            fig, ax = plt.subplots(ncols=2)
            norm = simple_norm(cutout.data, 'log', percent=99.5)
            ax[0].imshow(cutout.data, origin='lower', norm=norm)
            ax[0].plot(xcnew-x0, ycnew-y0, 'k+')
            ax[0].axis('off')
            ax[0].set_title('xc=%.2f, yc=%.2f\nmag=%.2f, FWHM=%.2f'%(xc,yc,mag,meanFWHM))

            res = star.compute_residual_image(epsf_init)/flux
            norm = simple_norm(res, min_cut=-1e-4, max_cut=1e-4)

            ax[1].imshow(res, origin='lower', norm=norm)
            ax[1].axis('off')
            ax[1].set_title('Residual')
            plt.tight_layout()
            plt.show()

            mean, med, std = sigma_clipped_stats(res)
            localback = med

        flag = input('Select this one? Y/N: ')
        if flag in 'Yy':
            #finalselect_index.append(index)
            finalselect_x.append(xcnew)
            finalselect_y.append(ycnew)
            fluxlist.append(flux)
            localbacklist.append(localback)

    tbl = Table({'x': finalselect_x, 'y': finalselect_y,\
                 'flux': fluxlist, 'bkg': localbacklist})

    if len(tbl)>0:
        tbl['ImageName'] = imagename
        tbl['select'] = 1
    else:
        tbl['ImageName'] = []
        tbl['select'] = []
    return tbl

'''
Functions for epsfstarlist manipulation
'''
def extract_epsfstarlist(startable, size, margin):
    datalist = []
    tablelist=  []

    for index in range(len(startable)):
        filename = startable['ImageName'][index]
        starx = startable['x'][index]
        stary = startable['y'][index]
        startbl_single = Table({'x': [starx], 'y': [stary]})

        data = fits.getdata(filename, 1)
        nddata = NDData(data)

        datalist.append(nddata)
        tablelist.append(startbl_single)

    psfstars = extract_stars(datalist, tablelist, size=size+2*margin)

    for index, star in enumerate(psfstars):
        # assign flux, weights, centers for the stars
        star.flux = startable['flux'][index]
        sigma = cutout_image(startable['ImageName'][index],\
                [startable['x'][index], startable['y'][index]],\
                star.data.shape, n_ext=2).data
        weights = 1/sigma**2
        star.weights = weights

    return psfstars

def update_epsfstarlist(origtblfile, origstarsfile, addtblfile, addstarsfile,
                        newtblfile, newstarsfile, epsf, epsf_error, size, margin):
    origtbl = Table.read(origtblfile)
    origstars = pickle.load(open(origstarsfile, 'rb'))
    addtbl = Table.read(addtblfile)
    addstars = pickle.load(open(addstarsfile, 'rb'))

    keepstar_add, newpsfstars_add = inspect_psfstars(addstars, epsf, epsf_error, size, margin)

    # update the table and psf stars
    newtbl = vstack([origtbl, addtbl[keepstar_add]])
    newstars = EPSFStars(origstars._data + newpsfstars_add._data)

    newtbl.write(newtblfile, overwrite=True)
    pickle.dump(newstars, open(newstarsfile, 'wb'))

    return newtbl, newstars

def inspect_psfstars(psfstars, epsf, epsf_error, size, margin):
    keepstar = []
    newpsfstars = []

    for index, star in enumerate(psfstars._data):
        print(index)
        res = star.compute_residual_image(epsf)[margin:-margin,margin:-margin]
        sigma = np.sqrt((1/star.weights/star.flux**2 + epsf_error**2))

        try:
            sigma = np.sqrt((1/star.weights/star.flux**2 + epsf_error**2))
        except:
            keepstar.append(False)
            continue

        res = res / star.flux / sigma[margin:-margin,margin:-margin]
        starimg = star.data[margin:-margin,margin:-margin]

        norm_star = simple_norm(starimg, 'log', percent=95)
        norm_res = simple_norm(res, min_cut=-3, max_cut=3)

        fig, axes = plt.subplots(ncols=2)
        axes[0].imshow(starimg, norm=norm_star)
        axes[1].imshow(res, norm=norm_res)

        for ax in axes:
            ax.axis('off')
        plt.show()

        flag = input('Keep the star? Y/N')
        if flag in 'Yy':
            keepstar.append(True)
            newpsfstars.append(star)
        else:
            keepstar.append(False)

    newpsfstars = EPSFStars(newpsfstars)
    return keepstar, newpsfstars

'''
Functions for building PSF
'''
def build_epsf_onestep(psfstars, size, margin, init_epsf):

    sc = SigmaClip(sigma=3, sigma_lower=3, sigma_upper=3, maxiters=100, cenfunc='median', stdfunc='std', grow=False)    

    epsf_builder = EPSFBuilder(oversampling=1, shape=size+2*margin, maxiters=50, sigma_clip=sc,\
                                progress_bar=False, norm_radius=10, smoothing_kernel='quartic',\
                               recentering_func=centroid_com, recentering_boxsize=21, recentering_maxiters=100)
    epsf, fitted_stars = epsf_builder.build_epsf(psfstars, init_epsf=init_epsf)
    return epsf, fitted_stars

def make_qaplot(startable, fitted_stars, epsf, epsf_error, size, margin, output):
    # generate a QA plot
    ncols = 3
    nrows = int(fitted_stars.n_stars/ncols) + 1

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols*2, figsize=[15, 2+3*nrows])
    if nrows == 1:
        for index, star in enumerate(fitted_stars):

            filename = startable['ImageName'][index]
            visitstr = filename[filename.find('visit') + 5]

            starimg = star.data[margin:-margin,margin:-margin]
            norm_star = simple_norm(starimg, 'log', percent=95)
            axes[index*2].imshow(starimg, norm=norm_star, origin='lower', cmap='viridis')

            res = star.compute_residual_image(epsf)[margin:-margin,margin:-margin]
            norm_res = simple_norm(res, percent=99)
            axes[index*2+1].imshow(res, norm=norm_res, origin='lower', cmap='viridis')

            axes[index*2].set_title('Visit %s, (%.1f, %.1f)'%(visitstr, star.center[0], star.center[1]))
            axes[index*2+1].set_title('residual')

        for ax in axes:
            ax.axis('off')
    else:
        for index, star in enumerate(fitted_stars):
            row_idx = int(index / ncols)
            col_idx = index % ncols

            filename = startable['ImageName'][index]
            xc = startable['x'][index]
            yc = startable['y'][index]
            visitstr = filename[filename.find('visit') + 5]

            starimg = star.data[margin:-margin,margin:-margin]
            norm_star = simple_norm(starimg, 'log', percent=95)
            axes[row_idx][col_idx*2].imshow(starimg, norm=norm_star, origin='lower', cmap='viridis')

            res = star.compute_residual_image(epsf)[margin:-margin,margin:-margin]
            sigma = np.sqrt((1/star.weights/star.flux**2 + epsf_error**2))
            res = res / star.flux / sigma[margin:-margin,margin:-margin]

            norm_res = simple_norm(res, min_cut=-3, max_cut=3)
            axes[row_idx][col_idx*2+1].imshow(res, origin='lower', cmap='viridis', norm=norm_res)

#            axes[row_idx][col_idx*2].set_title('Visit %s, (%.1f, %.1f)'%(visitstr, xc, yc))
            axes[row_idx][col_idx*2].set_title('PSF Star #%d'%index)
            axes[row_idx][col_idx*2+1].set_title('residual ($\sigma$)')

        for axrow in axes:
            for ax in axrow:
                ax.axis('off')

    plt.tight_layout()
    plt.savefig(output)

def generate_epsf_error(fitted_stars, epsf):
    allresid = np.array([star.compute_residual_image(epsf)/star.flux\
                         for star in fitted_stars])
    allerr_random = np.array([star.weights**(-0.5)/star.flux for star in fitted_stars])

    allresid = []

    for index, star in enumerate(fitted_stars):
        xc, yc = star.cutout_center
        xc_int, yc_int = ((star.shape[1] - 1) / 2.0, (star.shape[0] - 1) / 2.0)
        shift_x = xc_int - xc
        shift_y = yc_int - yc

        from scipy import ndimage

        res = star.compute_residual_image(epsf)/star.flux
        err_rdm = star.weights**(-0.5)/star.flux

        shiftres = ndimage.shift(res, shift=[shift_y, shift_x], order=1)
        shifterr = ndimage.shift(err_rdm, shift=[shift_y, shift_x], order=1)

        allresid.append(shiftres**2-shifterr**2)

    meanvar, medvar, _ = sigma_clipped_stats(allresid, axis=0, sigma=3)
    meanvar = meanvar * len(fitted_stars) / (len(fitted_stars)-1)

    meanvar[meanvar<0] = 0
    stddev = np.sqrt(meanvar)

    return stddev

def build_psfstar_list(imagelist, fwhm_range, mag_range, cutoutsize, edge,\
                       output=None, newdetect=False, excludecoord=None, origpsf=None):

    allstartbls = []

    for filename in imagelist:
        print(filename)
        # this is the quasar's position
        header = fits.getheader(filename, 1)
        w = WCS(header)
        qp = w.world_to_pixel(excludecoord)

        # step 1: detect the source
        detectfile = filename[:-5] + '.cat.fits'
        if newdetect or (not os.path.exists(detectfile)):
            data = fits.getdata(filename, 1)
            daofind = DAOStarFinder(fwhm=5, threshold=0.1)
            sources = daofind(data)
            sources.write(detectfile, overwrite=True)

        else:
            sources = Table.read(detectfile)

        # step 2: automatically select some stars
        selected_index = preselect_psf(filename, sources, peaklim=[1, 5000], fwhm_range=fwhm_range, edge=edge)

        header = fits.getheader(filename, 1)
        magzp = -6.10 - 2.5* np.log10(header['PIXAR_SR'])

        # step 3: visual inspection
        if origpsf is not None:
            epsf_init = EPSFModel(fits.getdata(origpsf, 0), norm_radius=20)
            tbl_vis_inspect = visual_inspection(filename, sources, selected_index,\
                                    quasar_position=qp, epsf_init=epsf_init, cutout_size=cutoutsize,\
                                    maglim=mag_range, magzp=magzp)
        else:
            tbl_vis_inspect = visual_inspection(filename, sources, selected_index,\
                                    quasar_position=qp, cutout_size=cutoutsize, maglim=mag_range, magzp=magzp)

        # step 5: save
        tbl_vis_inspect.write(filename[:-5] + '.starlist.fits', overwrite=True)

        if len(tbl_vis_inspect)>0:
            allstartbls.append(tbl_vis_inspect)

    # step 6: build epsf
    stackstartbl = vstack(allstartbls)
    stackstartbl.write(output, overwrite=True)

    return stackstartbl

def build_epsf_looping(startbl, psfstars, size, margin, output=None, qaplot=None, inspect=True):

    epsf, fitted_stars = build_epsf_onestep(psfstars, size, margin, None)
    epsf_error = generate_epsf_error(fitted_stars, epsf)

    if inspect:
        keepstar, newstars = inspect_psfstars(fitted_stars, epsf, epsf_error, size, margin)
        newstartbl = startbl[keepstar]

        epsf, fitted_stars = build_epsf_onestep(newstars, size, margin, init_epsf=epsf)
        epsf_error = generate_epsf_error(fitted_stars, epsf)

    else:
        newstartbl = startbl
        newstars = fitted_stars

    if output is not None:
        epsf_data = epsf.data[margin:-margin,margin:-margin]
        epsf_data[epsf_data<0] = 0
        epsf_hdu = fits.PrimaryHDU(data=epsf_data)
        epsferr_hdu = fits.ImageHDU(data=epsf_error[margin:-margin,margin:-margin])

        epsf_hdulist = fits.HDUList([epsf_hdu, epsferr_hdu])
        epsf_hdulist.writeto(output, overwrite=True)
        print(epsf_data.shape)

    if qaplot is not None:
        make_qaplot(newstartbl, newstars, epsf, epsf_error, \
                    size, margin, qaplot)

    return epsf, epsf_error, newstartbl, newstars

'''
The following functions are for building individual PSFs
'''

def save_star_list_filter_mod(filt, mod):
    if filt=='F356W':
        suffix = 'v4'
        pixscale = 0.03
        size = 101
        margin = 20
        edge = 500
        mag_range = [18,21]
    elif filt in ['F200W', 'F115W']:
        suffix = 'fine'
        pixscale = 0.015
        size = 201
        margin = 40
        edge = 1000
        mag_range=[18.5, 21.5]

    imagelist_J0100 = ['/media/minghao/data/JWST/image_reduction/J0100+2802/mruari_crf/%s/stacks/stack_filter%s_visit%d_mod%s_%s.fits'%(filt, filt, v, mod, suffix) for v in [1,2,3,4]]
    imagelist_J0148 = ['/media/minghao/data/JWST/image_reduction/J0148+0600/mruari_crf/%s/stacks/stack_filter%s_visit%d_mod%s.fits'%(filt, filt, v, mod) for v in [1,2,3,4]]
    imagelist_J1030 = ['/media/minghao/data/JWST/image_reduction/J1030+0524/mruari_crf/%s/stacks/stack_filter%s_visit%d_mod%s.fits'%(filt, filt, v, mod) for v in [1,3,4]]
    imagelist_J1120 = ['/media/minghao/data/JWST/image_reduction/J1120+0641/mruari_crf/%s/stacks/stack_filter%s_visit%d_mod%s_%s.fits'%(filt, filt, v, mod, suffix) for v in [1,2,3,4]]
    imagelist_J1148 = ['/media/minghao/data/JWST/image_reduction/J1148+5251/mruari_crf/%s/stacks/stack_filter%s_visit%d_mod%s_%s.fits'%(filt, filt, v, mod, suffix) for v in [1,2,3,4]]
    imagelist_J159 = ['/media/minghao/data/JWST/image_reduction/J159-02/mruari_crf/%s/stacks/stack_filter%s_visit%d_mod%s.fits'%(filt, filt, v, mod) for v in [1,2]]

    imagelist = imagelist_J0100 + imagelist_J0148 + imagelist_J1030 + imagelist_J1120 + imagelist_J1148 + imagelist_J159
#    imagelist = [imagelist_J1148[0]]

    fwhm_range_dict = {'F115W': [(63.2-3*2.1)/15, (63.2+3*2.1)/15],\
                   'F200W': [(75.9-3*1.0)/15, (75.9+3*1.0)/15],\
                   'F356W': [(140-4*3)/30.0, (140+4*3)/30.0]}

    fwhm_range = fwhm_range_dict[filt]
    print(fwhm_range_dict)
    return 0
    excludecoord = SkyCoord([SkyCoord(ra=27.156836, dec=6.0055433, unit='deg'),
                  SkyCoord(ra=177.069355, dec=52.863968, unit='deg'),
                  SkyCoord.from_name('J010013.0278+280225.828'),\
                  SkyCoord.from_name('J112001.4653+064123.788'),\
                    SkyCoord.from_name('J103027.10+052455.157'),\
                    SkyCoord.from_name('J103654.19-023237.94')])

    epsfdir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/epsf'
    origpsf = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data_old/epsf/epsf_master_%s_mod%s.fits'%(filt,mod)
    startbl = epsfdir + '/%s_mod%s_starlist.fits'%(filt, mod)
    starsav = epsfdir + '/%s_mod%s_starlist.dat'%(filt, mod)

    build_psfstar_list(imagelist, fwhm_range, mag_range, size, edge, excludecoord=excludecoord,\
                      output=startbl, origpsf=origpsf)

    startable = Table.read(startbl)
    epsfstars = extract_epsfstarlist(startable, size=size, margin=margin)
    pickle.dump(epsfstars, open(starsav, 'wb'))

def build_epsf_filt_mod(filt, mod, size, margin, inspect=True):
    epsfdir = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/epsf'
    output = epsfdir + '/epsf_%s_mod%s.fits'%(filt, mod)    
    qaplot = epsfdir + '/epsf_%s_mod%s.pdf'%(filt, mod)    

    if inspect:
        startblfile = epsfdir + '/%s_mod%s_starlist.fits'%(filt, mod)
        starsavfile = epsfdir + '/%s_mod%s_starlist.dat'%(filt, mod)
    else:
        startblfile = epsfdir + '/%s_mod%s_starlist.vis.fits'%(filt, mod)
        starsavfile = epsfdir + '/%s_mod%s_starlist.vis.dat'%(filt, mod)

    startbl = Table.read(startblfile)
    starsav = pickle.load(open(starsavfile, 'rb'))

    epsf, epsf_error, newstartbl, newstars = \
        build_epsf_looping(startbl, starsav, size, margin, output=output, qaplot=qaplot, inspect=inspect)

    if inspect:
        newstartbl.write(startblfile[:-5]+'.vis.fits', overwrite=True)
        pickle.dump(newstars, open(starsavfile[:-4]+'.vis.dat', 'wb'))

def main():
    '''
    # what if we want to add some stars?
    addtable = Table.read(epsfdir+'/F356W_moda_starlist_add.fits')
    addstars = extract_epsfstarlist(addtable, size=101, margin=20)
    pickle.dump(addstars, open('epsfstars_2.sav', 'wb'))

    # try combining the two
    update_epsfstarlist(epsfdir+'/F356W_moda_starlist.fits', 'epsfstars_1.sav',\
                        epsfdir+'/F356W_moda_starlist_add.fits', 'epsfstars_2.sav',\
                        'combined.fits', 'combined.sav', epsf, epsf_error, 101, 20) 
    '''
    save_star_list_filter_mod('F356W', 'a', inspect=False)

if __name__=='__main__':
    filt, mod = 'F115W', 'b'

    save_star_list_filter_mod(filt, mod)
#    build_epsf_filt_mod(filt, mod, 201, 40, inspect=False)

#    build_epsf_filt_mod('F356W', 'b', 101, 20, inspect=False)
