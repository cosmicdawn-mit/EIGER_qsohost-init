import os, sys
import numpy as np
import matplotlib.pyplot as plt

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.nddata import Cutout2D
from astropy.visualization import simple_norm
from astropy.stats import sigma_clipped_stats

from photutils.aperture import aperture_photometry
from photutils.aperture import CircularAperture
from photutils.centroids import centroid_com

def cutout_image(imagename, position, size, output=None, n_ext=1):
    hdulist = fits.open(imagename)

    wcs = WCS(hdulist[n_ext].header)
    cutout = Cutout2D(hdulist[n_ext].data, position=position, size=size, wcs=wcs)

    if output is not None:
        hdu = fits.PrimaryHDU(data=cutout.data)
        hdu.writeto(output, overwrite=True)

    return cutout

def measure_center(imagename, xc, yc, size=21, centroid_func=centroid_com):
    cutout_img = cutout_image(imagename, [xc, yc], size=21, n_ext=1)
    cutout_nse = cutout_image(imagename, [xc, yc], size=21, n_ext=2)

    mask = (cutout_img.data==0)

    center_new = centroid_com(data=cutout_img.data, mask=mask)#,\
    x0, y0 = cutout_img.origin_original

    xcnew = x0 + center_new[0]
    ycnew = y0 + center_new[1]

    return xcnew, ycnew

class NIRCamImage(object):
    def __init__(self, imagename):
        self.imagename = imagename

        # get some parameters
        self.header = fits.open(self.imagename)[1].header
        self.pixscale = 3600 * self.header['CDELT1']
        self.zeropoint = -6.10 - 2.5 * np.log10(self.header['PIXAR_SR'])

    def make_galfit_cube(self, position, size, output=None, fluxscale=1):
        hdulist = fits.open(self.imagename)

        header0 = hdulist[0].header
        header1 = hdulist[1].header
        # sci
        wcs = WCS(hdulist['SCI'].header)
        cutout_sci = Cutout2D(hdulist['SCI'].data, position=position, size=size, wcs=wcs)
        header1.update(cutout_sci.wcs.to_header())

        # nse
        # assume the same wcs
        cutout_nse = Cutout2D(hdulist['ERR'].data, position=position, size=size, wcs=wcs)

        # mask
        data_sci = cutout_sci.data
        data_nse = cutout_nse.data

        data_bpm = np.zeros(data_sci.shape)
        data_bpm[(data_nse==0)|(data_sci==0)] = 1
        data_bpm[np.isnan(data_sci)] = 1
        data_bpm = np.array(data_bpm, dtype=int)
       
        all_bpm = data_bpm

        # generate the new data cube
        ohdu0 = fits.PrimaryHDU(header=header0)

        ohdu0.header['INPUT'] = self.imagename
        ohdu0.header['XPOS'] = position[0]
        ohdu0.header['YPOS'] = position[1]

        ohdu1 = fits.ImageHDU(data=data_sci, header=header1)
        ohdu2 = fits.ImageHDU(data=data_nse)
        ohdu3 = fits.ImageHDU(data=all_bpm)

        for h in [ohdu1, ohdu2, ohdu3]:
            h.header.update(cutout_sci.wcs.to_header())

        ohdulist = fits.HDUList([ohdu0, ohdu1, ohdu2, ohdu3])
        if output is not None:
            ohdulist.writeto(output, overwrite=True)

        return ohdulist

def calculate_flux_scale(image1, pos1, image2, pos2, radius=10):
    aper1 = CircularAperture(pos1, r=radius)
    aper2 = CircularAperture(pos2, r=radius)

    phot1 = aperture_photometry(image1, aper1)
    phot2 = aperture_photometry(image2, aper2)

    flux1 = phot1['aperture_sum'][0]
    flux2 = phot2['aperture_sum'][0]

    return flux1/flux2

def prepare_one_quasar(coord, direct, output_dir, filt, size, visits=[1,2,3,4]):

    for visit in visits:
        if visit in [1, 2]:
            mod = 'a'
        else:
            mod = 'b'

        if not os.path.exists(output_dir):
            os.system('mkdir -p %s'%output_dir)

        if filt=='F356W':
            imagename = direct + '/%s/stacks/stack_filter%s_visit%d_mod%s_v4.fits'%(filt, filt, visit, mod)
        else:
            imagename = direct + '/%s/stacks/stack_filter%s_visit%d_mod%s_fine.fits'%(filt, filt, visit, mod)

        if not os.path.exists(imagename):
            imagename = direct + '/%s/stacks/stack_filter%s_visit%d_mod%s.fits'%(filt, filt, visit, mod)

        header0 = fits.getheader(imagename, 0)
        header = fits.getheader(imagename, 1)
        w = WCS(header)
        xc, yc = w.world_to_pixel(coord)
        xc = float(xc)
        yc = float(yc)

        newpos = measure_center(imagename, xc, yc)
        if np.any(np.isnan(newpos)) or np.any(np.isinf(newpos)):
            newpos = (xc, yc)

        output = output_dir + '/qso_filter%s_visit%d_mod%s.fits'%(filt, visit, mod)
        ncimg = NIRCamImage(imagename=imagename)
        cube = ncimg.make_galfit_cube(position=newpos, size=size, output=output, fluxscale=1)

def prepare_stars(startbl, size, output_dir):
    if not os.path.exists(output_dir):
        os.system('mkdir -p %s'%output_dir)

    for index in range(len(startbl)):
        filename = startbl['ImageName'][index]
        starx = startbl['x'][index]
        stary = startbl['y'][index]

        pos = [starx, stary]
        output = output_dir + '/star%d.fits'%index

        ncimg = NIRCamImage(imagename=filename)
        cube = ncimg.make_galfit_cube(position=pos, size=size, output=output)

def main():
    # use WCS
    coord_J0148 = SkyCoord(ra=27.156836, dec=6.0055433, unit='deg')
    coord_J1148 = SkyCoord(ra=177.069355, dec=52.863968, unit='deg')
    coord_J0100 = SkyCoord.from_name('J010013.0278+280225.828')
    coord_J1120 = SkyCoord.from_name('J112001.4649+064123.785')
    coord_J1030 = SkyCoord.from_name('J103027.10+052455.157')
    coord_J159 = SkyCoord.from_name('J103654.19-023237.94')

    direct_J0148 = '/media/minghao/data/JWST/image_reduction/J0148+0600/mruari_crf/'
    direct_J1148 = '/media/minghao/data/JWST/image_reduction/J1148+5251/mruari_crf/'
    direct_J0100 = '/media/minghao/data/JWST/image_reduction/J0100+2802/mruari_crf/'
    direct_J1120 = '/media/minghao/data/JWST/image_reduction/J1120+0641/mruari_crf/'
    direct_J1030 = '/media/minghao/data/JWST/image_reduction/J1030+0524/mruari_crf/'
    direct_J159 = '/media/minghao/data/JWST/image_reduction/J159-02/mruari_crf/'

    outdir_J159 = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/quasars/J159-02/'
    outdir_J0100 = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/quasars/J0100+2802/'
    outdir_J0148 = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/quasars/J0148+0600/'
    outdir_J1030 = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/quasars/J1030+0524/'
    outdir_J1120 = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/quasars/J1120+0641/'
    outdir_J1148 = '/home/minghao/Research/Projects/JWST/qsohost_imaging/data/quasars/J1148+5251/'

#    prepare_one_quasar(coord_J159, direct_J159, outdir_J159, filt='F356W', size=102, visits=[1,2])
#    prepare_one_quasar(coord_J159, direct_J159, outdir_J159, filt='F200W', size=202, visits=[1,2])
#    prepare_one_quasar(coord_J159, direct_J159, outdir_J159, filt='F115W', size=202, visits=[1,2])

#    prepare_one_quasar(coord_J0100, direct_J0100, outdir_J0100, filt='F356W', size=102, visits=[1,2,3,4])
#    prepare_one_quasar(coord_J0100, direct_J0100, outdir_J0100, filt='F200W', size=202, visits=[1,2,3,4])
#    prepare_one_quasar(coord_J0100, direct_J0100, outdir_J0100, filt='F115W', size=202, visits=[1,2,3,4])

#    prepare_one_quasar(coord_J0148, direct_J0148, outdir_J0148, filt='F356W', size=102, visits=[1,2,3,4])
#    prepare_one_quasar(coord_J0148, direct_J0148, outdir_J0148, filt='F200W', size=202, visits=[1,2,3,4])
#    prepare_one_quasar(coord_J0148, direct_J0148, outdir_J0148, filt='F115W', size=202, visits=[1,2,3,4])

#    prepare_one_quasar(coord_J1030, direct_J1030, outdir_J1030, filt='F356W', size=102, visits=[1,3,4])
#    prepare_one_quasar(coord_J1030, direct_J1030, outdir_J1030, filt='F200W', size=202, visits=[1,3,4])
#    prepare_one_quasar(coord_J1030, direct_J1030, outdir_J1030, filt='F115W', size=202, visits=[1,3,4])

    prepare_one_quasar(coord_J1120, direct_J1120, outdir_J1120, filt='F356W', size=102, visits=[1,2,3,4])
    prepare_one_quasar(coord_J1120, direct_J1120, outdir_J1120, filt='F200W', size=202, visits=[1,2,3,4])
    prepare_one_quasar(coord_J1120, direct_J1120, outdir_J1120, filt='F115W', size=202, visits=[1,2,3,4])

#    prepare_one_quasar(coord_J1148, direct_J1148, outdir_J1148, filt='F356W', size=102, visits=[1,2,3,4])
#    prepare_one_quasar(coord_J1148, direct_J1148, outdir_J1148, filt='F200W', size=202, visits=[1,2,3,4])
#    prepare_one_quasar(coord_J1148, direct_J1148, outdir_J1148, filt='F115W', size=202, visits=[1,2,3,4])

if __name__=='__main__':
    main()
