from math import log
from os import chdir
import os
import re, sys, time
import numpy as np

## Third party

from astropy.cosmology import Planck18 as cosmo
from astropy.io.fits import getdata
from astropy.io import fits
from astropy.table import Table
from astropy.units import (ABmag, Angstrom, mgy, nJy, Quantity,
                           spectral_density, zero_point_flux,erg, cm, second)
from astropy.visualization import quantity_support
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
from numpy import loadtxt, r_
from scipy.interpolate import interp1d

print("importing prospector libraries")
from prospect.io.read_results import results_from
from prospect.models.priors import LogUniform, Normal, Uniform, ClippedNormal, TopHat
from prospect.models.sedmodel import SedModel, LineSpecModel
from prospect.models.templates import TemplateLibrary
from prospect.sources import CSPSpecBasis
from prospect import prospect_args
from prospect.fitting import fit_model, lnprobfn
from prospect.io import write_results as writer
import prospect.io.read_results as reader
from prospect.utils.obsutils import fix_obs
from sedpy.observate import Filter, load_filters

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)

import matplotlib as mpl
import matplotlib.ticker as ticker
mpl.rcParams.update(mpl.rcParamsDefault)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "Computer Modern Serif",
})

plt.rcParams["ytick.direction"] = 'in'
plt.rcParams["xtick.direction"] = 'in'

import astropy.units as u

def fit_mags_host(mags, magerrs, redshift, savefile, return_model_only=False):

    # load filters
    if len(mags)==2:
        filters = load_filters(['jwst_f200w', 'jwst_f356w'])
    elif len(mags)==3:
        filters = load_filters(['jwst_f115w', 'jwst_f200w', 'jwst_f356w'])

    # convery mags to maggies
    maggies = 10**(-0.4 * np.array(mags))
    maggies_unc = np.array(magerrs) * maggies / 1.086

    # build the observation
    
    obs = dict(wavelength=None, spectrum=None, unc=None, redshift=redshift,
           maggies=maggies, maggies_unc=maggies_unc, filters=filters)

    obs = fix_obs(obs)

    # build the model
    this_hubbletime = -1 * (cosmo.lookback_time(redshift)-cosmo.lookback_time(50)).value

#    model_params = TemplateLibrary['parametric_sfh'] | TemplateLibrary['igm'] | TemplateLibrary['nebular']

    # IMF type 1 = Charbrier, 2 = Kroupa....JAGUAR uses Charbrier                
#    model_params['imf_type'] = {'init': 1}
#    model_params['masscut'] = {'init': 100}
#    model_params['mass']['prior'] = LogUniform(mini=10**7, maxi=10**13)
#    model_params['mass']['init'] = 1e9

#    model_params['logzsol']['isfree'] = True
#    model_params['logzsol']['init'] = -0.7

#    model_params['dust2']['isfree'] = True
#    model_params['dust2']['init'] = 0.5

#    model_params['dust_type']['init'] = 0

#    model_params['tage']['prior'] = Uniform(mini=0.001, maxi=this_hubbletime)
#    model_params['tau']['prior'] = TopHat(mini=0.1, maxi=20)            
#    model_params['tau']['isfree'] = False
#    model_params['tau']['init'] = 1


#    model_params['zred']['isfree'] = False
#    model_params['zred']['init'] = redshift

    model_params = TemplateLibrary['parametric_sfh'] | TemplateLibrary['igm'] | TemplateLibrary['nebular']
    model_params['dust_type']['init'] = 2
    model_params['dust1'] = dict(N=1, init=0.0, isfree=False)

    # IMF type 1 = Charbrier, 2 = Kroupa....JAGUAR uses Charbrier                
    model_params['imf_type'] = {'init': 1}
    model_params['masscut'] = {'init': 100}
    model_params['mass']['prior'] = LogUniform(mini=10**8, maxi=10**12)
    model_params['mass']['init'] = 5e9

    model_params['logzsol']['isfree'] = True
    model_params['logzsol']['init'] = -1.0

    model_params['tage']['prior'] = Uniform(mini=0.001, maxi=this_hubbletime)
    model_params['tau']['prior'] = TopHat(mini=0.1, maxi=20)            

#    model_params['linespec_scaling'] = dict(N=1, init=1.0, isfree=True, prior=TopHat(mini=0.1,maxi=1.0))

    model_params['zred']['isfree'] = False
    model_params['zred']['init'] = redshift

    # add in emission lines
    model_params["gas_logu"]["isfree"] = True
    model_params["gas_logu"]['prior'] = TopHat(mini=-3,maxi=1)
    model_params["gas_logz"]["isfree"] = True

    for key, item in model_params.items():
        print(key, item)

    print("Model params programmed in")
    print("Building the SED model")
    sed_model = SedModel(model_params)

#    print(model_params)
#    return 0

    if return_model_only:
        return (None, obs, sed_model)

    sps = CSPSpecBasis()

    noise_model = (None, None)
    
    fitting_kwargs = dict(nlive_init=400, nested_method="rwalk", nested_target_n_effective=1000, nested_dlogz_init=0.05)
    output = fit_model(obs, sed_model, sps, noise=noise_model)
    
    writer.write_hdf5(savefile, {}, sed_model, obs,
                 output["sampling"][0], None,
                 sps=sps,
                 tsample=output["sampling"][1],
                 toptimize=0.0)

    return output, obs, sed_model


def plot_results(mags, magerrs, savefile, title, output):

    out, out_obs, out_model = reader.results_from(savefile)

    # plotting SED

    sfig, saxes = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[1, 4]), sharex=True)
    ax = saxes[1]
    pwave = np.array([f.wave_effective for f in out_obs["filters"]])
    # plot the data
    ax.plot(pwave, out_obs["maggies"], linestyle="", marker="o", color="k")
    ax.errorbar(pwave,  out_obs["maggies"], out_obs["maggies_unc"], linestyle="", color="k", zorder=10)
    ax.set_ylabel(r"$f_\nu$ (maggies)")
    ax.set_xlabel(r"$\lambda$ (AA)")
    ax.set_xlim(1e4, 4e4)
    ax.set_ylim(out_obs["maggies"].min() * 0.1, out_obs["maggies"].max() * 5)
    ax.set_yscale("log")

    # get the best-fit SED
    bsed = out["bestfit"]
    ax.plot(bsed["restframe_wavelengths"] * (1+out_obs["redshift"]), bsed["spectrum"], color="firebrick", label="MAP sample")
    ax.plot(pwave, bsed["photometry"], linestyle="", marker="s", markersize=10, mec="orange", mew=3, mfc="none")

    ax = saxes[0]
    chi = (out_obs["maggies"] - bsed["photometry"]) / out_obs["maggies_unc"]
    ax.plot(pwave, chi, linestyle="", marker="o", color="k")
    ax.axhline(0, color="k", linestyle=":")
    ax.set_ylim(-2, 2)
    ax.set_ylabel(r"$\chi_{\rm best}$")
    ax.set_title(title)

    plt.savefig(output+'_sed.pdf')
    plt.close('all')

    from prospect.plotting import corner
    nsamples, ndim = out["chain"].shape
    cfig, axes = plt.subplots(ndim, ndim, figsize=(10,9))
    axes = corner.allcorner(out["chain"].T, out["theta_labels"], axes, weights=out["weights"], color="royalblue", show_titles=True)

    from prospect.plotting.utils import best_sample
    pbest = best_sample(out)
    corner.scatter(pbest[:, None], axes, color="firebrick", marker="o")

    cfig.suptitle(title)
    plt.savefig(output+'_corner.pdf')


def plot_results_talk(mags, magerrs, savefile, title, output, model):

    out, out_obs, out_model = reader.results_from(savefile)

    import random
    chain_sample = random.choices(out['chain'], out['weights'], k=1000)

    sps = CSPSpecBasis()

    try:
        lowspec, medspec, highspec = np.loadtxt('./%s_modelspec_save.txt'%output)
        
    except:
        speclist = []
        for theta in chain_sample:
            spec, phot, mfrac = model.predict(theta, obs=out['obs'], sps=sps)
            speclist.append(spec)

        lowspec, medspec, highspec = np.percentile(speclist, [16, 50, 84], axis=0)

        np.savetxt('./%s_modelspec_save.txt'%output, [lowspec, medspec, highspec])

    # plotting SED

    fig, ax = plt.subplots(figsize=[5.2,4])
    pwave = np.array([f.wave_effective for f in out_obs["filters"]]) / 1e4
    swave = sps.wavelengths * (1+out_obs["redshift"]) / 1e4

    mask = (swave>1)&(swave<4)

    # plot the data
#    ax.plot(pwave, out_obs["maggies"], linestyle="", marker="o", color="steelblue")
    fluxfactor = 3631*1e6
    ax.errorbar(pwave,  fluxfactor*out_obs["maggies"], out_obs["maggies_unc"]*fluxfactor, marker='o', linestyle="",\
                color="red", zorder=10, label='NIRCam Imaging')
    ax.plot(swave[mask], medspec[mask]*fluxfactor, 'k-', label='Prospector Model')
    ax.fill_between(swave[mask], y1=lowspec[mask]*fluxfactor, y2=highspec[mask]*fluxfactor, color='k', alpha=0.2)

    ax.set_ylabel(r"$\mathrm{F_\nu~[\mu Jy]}$", fontsize=12)
    ax.set_xlabel(r"Wavelengths [micron]", fontsize=12)
    ax.set_xlim(1, 4)
    ax.set_ylim(out_obs["maggies"].min() * 0.5*fluxfactor, out_obs["maggies"].max() * 5*fluxfactor)
    ax.set_yscale("log")
    ax.set_title('%s'%output, fontsize=14)

    ax.tick_params(axis="both", direction="in")
    ax.tick_params(axis="both", direction="in", which='minor')

    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output+'_sed.pdf')
    plt.show()
    plt.close('all')


def get_UVmag(mags, magerrs, savefile, title, output, model):
    out, out_obs, out_model = reader.results_from(savefile)

    import random
    chain_sample = random.choices(out['chain'], out['weights'], k=1000)

    z = out_obs['redshift']

    sps = CSPSpecBasis()
    swave = sps.wavelengths# * (1+out_obs["redshift"]) / 1e4

    Muvlist = []
    Moptlist = []
    for theta in chain_sample:
        spec, phot, mfrac = model.predict(theta, obs=out['obs'], sps=sps)
        f1500 = np.interp(1500, swave, spec)
        f1500_abs = f1500 * (cosmo.luminosity_distance(z).to('pc') / 10 / u.pc)**2 / (1+z)
        f4800 = np.interp(4800, swave, spec)
        f4800_abs = f4800 * (cosmo.luminosity_distance(z).to('pc') / 10 / u.pc)**2 / (1+z)

        Muv = -2.5 * np.log10(f1500_abs.value)
        Mopt = -2.5 * np.log10(f4800_abs.value)

        Muvlist.append(Muv)
        Moptlist.append(Mopt)

    low, med, high =  np.percentile(Muvlist, [16, 50, 84])
    plt.hist(Muvlist)
    plt.title('Muv=%.2f-%.2f+%.2f'%(med, med-low, high-med))
    plt.savefig('Muv_%s.pdf'%title)
    plt.show()

    low, med, high =  np.percentile(Moptlist, [16, 50, 84])
    plt.hist(Moptlist)
    plt.title('Mopt=%.2f-%.2f+%.2f'%(med, med-low, high-med))
    plt.savefig('Mopt_%s.pdf'%title)
    plt.show()


def plot_luminosity_size_relation():
    mags = [-22.84, -22.73]
    magerrs_p = [0.15, 0.08]
    magerrs_m = [0.16, 0.09]

    zlist = [5.98, 7.08]
    ad = cosmo.angular_diameter_distance(zlist).to('kpc').value

    radii_kpc = (np.array([0.41, 0.27])/206265 * ad)
    radii_err = np.array([0.02, 0.02])/206265 * ad

    logradii_kpc = np.log10(radii_kpc)
    logradii_err = radii_err / radii_kpc / np.log(10)

    fig, ax = plt.subplots(figsize=[5,4])
    ax.errorbar(x=mags, y=logradii_kpc, xerr=[magerrs_m, magerrs_p], yerr=logradii_err,\
                marker='o', color='r', ls='', label='This Work')

    # shibuya+15
    Muvlist = np.arange(-23.5, -17, 0.1)
    r0_z65 = 6.9 * (1+6.5) ** (-1.2)
    fluxratio = 10**(-0.4*(Muvlist+21))
    relist = r0_z65 * (fluxratio)**0.27
    logrelist = np.log10(relist)
#    ax.plot(Muvlist, logrelist, 'k-', label='Shibuya+15')
#    ax.fill_between(Muvlist, y1=logrelist-0.6/np.log(10), y2=logrelist+0.6/np.log(10), color='k', alpha=0.2)

    # yang+22
    r0_yang = 10**(-0.22)
    fluxratio = 10**(-0.4*(Muvlist+21))
    relist_yang = r0_yang * (fluxratio)**0.2
    logrelist_yang = np.log10(relist_yang)

    ax.plot(Muvlist, logrelist_yang, 'k-', label='Yang+22 ($z\sim8$)')
    ax.fill_between(Muvlist, y1=logrelist_yang-0.53/np.log(10), y2=logrelist_yang+0.53/np.log(10), color='k', alpha=0.2)

    # ding+22
    radii_ding_kpc = np.array([1.9, 0.8])
    radii_ding_err = np.array([1.1, 0.1])
    Muv_ding = np.array([-18.75, -20.86])
    Muv_ding_u = np.array([-18.05, -20.62])
    Muv_ding_l = np.array([-19.47, -20.97])

    logradii_ding_kpc = np.log10(radii_ding_kpc)
    logradii_ding_err = radii_ding_err / radii_ding_kpc / np.log(10)

    ax.errorbar(x=Muv_ding, y=logradii_ding_kpc, xerr=[Muv_ding-Muv_ding_l, Muv_ding_u-Muv_ding], yerr=logradii_ding_err,\
                marker='^', color='steelblue', ls='', label='Ding+22')


    ax.set_xlim([-23.5, -18])

    ax.set_xlabel(r'$M_{UV}$', fontsize=12)
    ax.set_ylabel(r'$\log R_e$ [kpc]', fontsize=12)

    plt.legend(frameon=False, fontsize=12)
    plt.tight_layout()
    plt.savefig('MLR.pdf')
    plt.show()

# utils
def find_next_filename(filename, findnext=True):
    for index in range(1000):
        thisfilename = filename[:-3]+'_%d.h5'%index
        if os.path.exists(thisfilename):
            continue
        else:
            break
    if findnext:
        return thisfilename
    else:
        thisfilename = filename[:-3]+'_%d.h5'%(index-1)
        return thisfilename

def plot_sed_axes(ax, mags, magerrs, savefile, qname, model, title):
    maggies = 10**(-0.4 * np.array(mags))
    maggies_unc = np.array(magerrs) * maggies / 1.086

    out, out_obs, out_model = reader.results_from(savefile)

    import random
    chain_sample = random.choices(out['chain'], out['weights'], k=1000)

    sps = CSPSpecBasis()

    try:
        lowspec, medspec, highspec = np.loadtxt('./%s_modelspec_save.txt'%qname)
        
    except:
        print('no save file found')
        speclist = []
        for theta in chain_sample:
            spec, phot, mfrac = model.predict(theta, obs=out['obs'], sps=sps)
            speclist.append(spec)

        lowspec, medspec, highspec = np.percentile(speclist, [16, 50, 84], axis=0)

        np.savetxt('./%s_modelspec_save.txt'%qname, [lowspec, medspec, highspec])

    # plotting SED
    filters = load_filters(['jwst_f115w', 'jwst_f200w', 'jwst_f356w'])

    pwave = np.array([f.wave_effective for f in filters]) / 1e4
    pwidth = np.array([f.rectangular_width/2 for f in filters]) / 1e4
    swave = sps.wavelengths * (1+out_obs["redshift"]) / 1e4# this is micron

    mask = (swave>0.7)&(swave<4.5)

    # plot the data
    fluxfactor = 3631*1e6

    Fnu_Jy = medspec * 3631
    Fnu_cgs = Fnu_Jy * 1e-23
    Flambda_cgs = Fnu_cgs * 3e10 / (swave*1e-4)**2
    Fnu_Jy_low = lowspec * 3631
    Fnu_cgs_low = Fnu_Jy_low * 1e-23
    Flambda_cgs_low = Fnu_cgs_low * 3e10 / (swave*1e-4)**2

    # test magnitude
    filt = Filter('jwst_f115w')

    ax.errorbar(pwave[1:],  fluxfactor*maggies[1:], yerr=maggies_unc[1:]*fluxfactor, xerr=pwidth[1:], marker='o', linestyle="",\
                color="red", zorder=10, label='NIRCam Imaging', markersize=10)
    ax.errorbar(pwave[:1],  fluxfactor*maggies[:1], yerr=maggies_unc[:1]*fluxfactor, xerr=pwidth[:1], marker='o', linestyle="",\
                color="red", zorder=10, mfc='none', markersize=10)

    ax.plot(swave[mask], medspec[mask]*fluxfactor, 'k-', label='Prospector Model')
    ax.fill_between(swave[mask], y1=lowspec[mask]*fluxfactor, y2=highspec[mask]*fluxfactor, color='k', alpha=0.2)

    ax.set_xlim(0.8,4.5)
    ax.set_ylim(out_obs["maggies"].min() * 0.5*fluxfactor, out_obs["maggies"].max() * 5*fluxfactor)
    ax.set_yscale("log")
    ax.set_title('%s'%title, fontsize=14)

    np.savetxt('./%s_modelwave_save.txt'%qname, swave)

    return ax

def get_scale(m_F356W, redshift):
    # load the J0148 template
    sps = CSPSpecBasis()
    swave_J0148 = sps.wavelengths * (1+5.977)
    swave_target = sps.wavelengths * (1+redshift)

    savefile = './save/J0148_sedfitting_0.h5'
    lowspec, medspec, highspec = np.loadtxt('./J0148+0600_modelspec_save.txt')
    # these are maggies. changing to F_lambda

    Fnu_Jy = medspec * 3631
    Fnu_cgs = Fnu_Jy * 1e-23
    Flambda_cgs = Fnu_cgs * 3e10 / (swave_J0148*1e-8)**2
#    plt.plot(swave_J0148, Fnu_Jy)
#    plt.show()
    # test magnitude
    filt = Filter('jwst_f356w')
#    print(filt.ab_mag(swave_J0148, Flambda_cgs/1e8))

    # now convert
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    lumdist_J0148 = cosmo.luminosity_distance(5.977).to('kpc').value
    lumdist = cosmo.luminosity_distance(redshift).to('kpc').value

    Llambda = Flambda_cgs * (1+5.977) * 4 * np.pi * lumdist_J0148**2
    Flambda_new = Llambda / (1+redshift) / (4 * np.pi * lumdist**2)

    newmag = filt.ab_mag(swave_target, Flambda_new/1e8)

    factor = 10**(-0.4*(m_F356W-newmag))
    print(np.log10(factor), m_F356W-newmag)
    
    return factor

def plot_J0148_sed():
    # load data
    mags_J0148 = [23.48, 23.51, 22.61]
    magerrs_J0148 = [0.24, 0.15, 0.07]
    redshift_J0148 = 5.977
    savefile = './save/J0148_sedfitting_0.h5'

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[5,3.5])

    _, _, model = fit_mags_host(mags_J0148[1:], magerrs_J0148[1:], redshift_J0148,\
                                savefile, return_model_only=True)
    title = 'J0148+0600 SED Fitting'
    print('start plotting')
    ax = plot_sed_axes(ax, mags_J0148, magerrs_J0148, savefile, 'J0148+0600', model, \
                       title=r'Prospector: $\log M_* [M_\odot]=10.74^{+0.31}_{-0.30}$')

    ax.set_ylabel(r"$\mathrm{F_\nu~[\mu Jy]}$", fontsize=16)
    ax.set_xlabel(r"Wavelengths [micron]", fontsize=16)

#    ax.legend(fontsize=14, loc='upper left')

    plt.tight_layout()
    plt.savefig('./sedfitting_J0148_proposal.pdf')
    plt.show()

def plot_J159_sed():
    # load data
    mags_J0148 = [24.83, 24.82, 23.98]
    magerrs_J0148 = [0.06, 0.23, 0.16]
    redshift_J0148 = 6.381
    savefile = './save/J159_sedfitting_0.h5'

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[5,3.5])

    _, _, model = fit_mags_host(mags_J0148[1:], magerrs_J0148[1:], redshift_J0148,\
                                savefile, return_model_only=True)
    title = 'J0148+0600 SED Fitting'
    print('start plotting')
    ax = plot_sed_axes(ax, mags_J0148, magerrs_J0148, savefile, 'J159-02', model, \
                       title=r'Prospector: $\log M_* [M_\odot]=10.14^{+0.34}_{-0.36}$')

    ax.set_ylabel(r"$\mathrm{F_\nu~[\mu Jy]}$", fontsize=16)
    ax.set_xlabel(r"Wavelengths [micron]", fontsize=16)

#    ax.legend(fontsize=14, loc='upper left')

    plt.tight_layout()
    plt.savefig('./sedfitting_J159_proposal.pdf')
    plt.show()


def plot_all_seds():

    # load data
    mags_J0148 = [23.48, 23.51, 22.61]
    magerrs_J0148 = [0.24, 0.15, 0.07]
    mags_J159 = [24.83, 24.82, 23.98]
    magerrs_J159 = [0.06, 0.23, 0.16]
    mags_J1120 = [24.78, 24.43, 24.45]
    magerrs_J1120 = [0.12, 0.10, 0.20]

    magslist = [mags_J0148, mags_J159, mags_J1120]
    magerrslist = [magerrs_J0148, magerrs_J159, magerrs_J1120]
    savefilelist = ['./save/J0148_sedfitting_0.h5',\
                    './save/J159_sedfitting_0.h5',\
                    './save/J1120_sedfitting_1.h5']
    zlist = [5.977,6.381, 7.085]
    titlelist = ['J0148+0600', 'J159-02', 'J1120+0641']

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=[12,4])

    for index, ax in enumerate(axes):

        mags = magslist[index]
        magerrs = magerrslist[index]
        savefile = savefilelist[index]
        redshift = zlist[index]
        _, _, model = fit_mags_host(mags[1:], magerrs[1:], redshift, savefile, return_model_only=True)
        title = titlelist[index]
        print('start plotting')
        ax = plot_sed_axes(ax, mags, magerrs,savefile, title, model, title)

    fig.supylabel(r"$\mathrm{F_\nu~[\mu Jy]}$", fontsize=16)
    fig.supxlabel(r"Wavelengths [micron]", fontsize=16)

    axes[0].legend(fontsize=14)

    plt.tight_layout()
#    plt.savefig('./sedfitting.pdf')
    plt.show()

    # print info for J1120 F115W

def run_all_fitting(qsoname):
    # J0148
    if qsoname=='J0148+0600':
        mags = [23.48, 23.51, 22.61]
        magerrs = [0.24, 0.15, 0.07]
        savefile = './save/J0148_sedfitting_3bands.h5'
        thissavefile = find_next_filename(savefile, findnext=True)
        _, _, model = fit_mags_host(mags, magerrs, 5.977, thissavefile, return_model_only=False)
        plot_results(mags, magerrs, thissavefile, 'J0148+0600', 'J0148+0600')

    #    plot_results_talk(mags, magerrs, thissavefile, 'J0148+0600', 'J0148+0600', model=model)
    #    get_UVmag(mags, magerrs, thissavefile, 'J0148+0600', 'J0148+0600', model=model)
        return 0

    elif qsoname=='J1120+0641':
        # J1120
        mags = [24.43, 24.45]
        magerrs = [0.10, 0.20]
        savefile = './save/J1120_sedfitting.h5'
        thissavefile = find_next_filename(savefile, findnext=False)
        _, _, model = fit_mags_host(mags, magerrs, 7.085, thissavefile, return_model_only=True)
        plot_results(mags, magerrs, thissavefile, 'J1120+0641', 'J1120+0641')

#        plot_results_talk(mags, magerrs, thissavefile, 'J1120+0641', 'J1120+0641', model=model)
#        get_UVmag(mags, magerrs, thissavefile, 'J1120+0641', 'J1120+0641', model=model)


    elif qsoname=='J159-02':
        # J159
        mags = [24.82, 23.98]
        magerrs = [0.23, 0.16]
        savefile = './save/J159_sedfitting.h5'
        thissavefile = find_next_filename(savefile, findnext=False)
        _, _, model = fit_mags_host(mags, magerrs, 6.381, thissavefile, return_model_only=True)
        plot_results(mags, magerrs, thissavefile, 'J159-02', 'J159-02')
#    plot_results_talk(mags, magerrs, thissavefile, 'J159-02', 'J159-02', model=model)

#    get_UVmag(mags, magerrs, thissavefile, 'J159-02', 'J159-02', model=model)


#    fit_mags_host(mags, magerrs, 7.08, thissavefile)
#    plot_results(mags, magerrs, thissavefile, 'J1120+0641', 'J1120+0641')
#    plot_luminosity_size_relation()


def print_logMs_and_error():
    Mslist = np.array([54717745689.86, 13985752405.24, 6462352136.57])
    uelist = np.array([56456094919.94, 16868136042.28, 4428057742.81])
    lelist = np.array([27313351694.79, 7930627424.33, 3324681869.92])

    logMs = np.log10(Mslist)
    logMs_ue = np.log10(Mslist+uelist) - np.log10(Mslist)
    logMs_le = np.log10(Mslist) - np.log10(Mslist-lelist)

    print(logMs)
    print(logMs_ue)
    print(logMs_le)

if __name__=='__main__':
#    plot_all_seds()
    plot_J0148_sed()
#    plot_J159_sed()

#    print_logMs_and_error()
    # J0100
#    get_scale(17.084+3.5, 6.327)
#    get_scale(19.39262+3.5, 6.304)
#    get_scale(18.72633+3.5, 6.422)
#    run_all_fitting('J0148+0600')
