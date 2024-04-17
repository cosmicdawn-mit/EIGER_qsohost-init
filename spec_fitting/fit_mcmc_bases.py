import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#import seaborn as sns
import astropy.units as u
import scipy.interpolate as interpol
from scipy.integrate import simps, trapezoid, simpson
import astropy.constants as const
from scipy.stats import binned_statistic_2d
from astropy.stats import sigma_clip
from astropy.table import Table
from astropy.io import fits
from astropy.convolution import convolve, Box1DKernel, Gaussian1DKernel
from astropy.coordinates import SkyCoord
from astropy.modeling.physical_models import BlackBody
import astropy.cosmology as cosmo
#import corner
import emcee
import pickle
import astropy.cosmology as cosmology
cosmo = cosmology.Planck18
import matplotlib.gridspec as gridspec
import warnings
#warnings.filterwarnings("ignore")
#import zeus

mgii = 2798.7
civ_a, civ_b = 1548., 1550.
lya, lyb = 1215.6701, 1025.7223
lyc, lyd, lye, lycont = 972.5368, 949.7, 937.8, 911.7
siiv_a, siiv_b = 1393, 1402
ciii = 1907 # CIII] emission line
nv_a, nv_b = 1238, 1242
siii_a, siii_b = 1260, 1304
hbeta = 4861.4
hgamma = 4340.5
hdelta = 4101.7
heii = 4685.7 
oiii_a, oiii_b = 4958.9, 5006.8 # wavelength in air not vacuum (Zakamska et al. 2003: https://arxiv.org/pdf/astro-ph/0309551.pdf)
oiii_c = 4363.2
fev_sii = 4072.5 #[FeV] + [SII]
neiii_he = 3967.8 # [NeIII] + H_epsilon

nu_rest_cii = 1900.548
nu_rest_co65 = 691.473
nu_rest_co54 = 576.267


def GetInitialPositionsBalls(nwalkers, best_guess, ndim, percent = 0.01):
    p0 = np.random.uniform(low=-1, high=1, size=(nwalkers, ndim))
    for i in range(0, ndim):
        scale = best_guess[i]
        if np.abs(scale)<1e-5:
            scale = 10
        p0[:, i] = best_guess[i] + p0[:, i] * percent * scale

    return p0

def GetInitialPositions(nwalkers, coef_min, coef_max, ndim):	
    p0 = np.random.uniform(size = (nwalkers, ndim))
    for i in range(0, ndim):
        p0[:, i] = coef_min[i] + p0[:, i] * (coef_max[i] - coef_min[i])
    return p0

def lnprob_jwst(theta, wl, flux, ivar, coef_min, coef_max, z, iron, modelfunc):
    lp = lnprior_jwst(theta, coef_min, coef_max)
    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike_jwst(wl, flux, ivar, theta, z, iron, modelfunc)

def lnprior_jwst(theta, coef_min, coef_max):
    
    if all(coef_min[i] < theta[i] < coef_max[i] for i in range(len(theta))):
        #if sigma_oiii_a2 > sigma_oiii_a1 and sigma_oiii_b2 > sigma_oiii_b1 and sigma_hbeta_2 > sigma_hbeta_1:
            return 0.0
    return -np.inf

def flux_model_no_iron(wl, theta, z, iron, components=False):

    a0, alpha, a_oiii_a1, sigma_oiii_a1, dv_oiii_a1, a_oiii_b1, sigma_oiii_b1, dv_oiii_b1, a_hbeta_1, sigma_hbeta_1, dv_hbeta_1, a_hbeta_2, sigma_hbeta_2, dv_hbeta_2 = theta

    model_pl = PowerLaw(a0, alpha, wl, z)
    model_iron = 0

    model_gauss_oiii_a1 = Gauss([a_oiii_a1, sigma_oiii_a1, dv_oiii_a1], wl, z, oiii_a)
    model_gauss_oiii_b1 = Gauss([a_oiii_b1, sigma_oiii_b1, dv_oiii_b1], wl, z, oiii_b)
    model_gauss_hbeta_1 = Gauss([a_hbeta_1, sigma_hbeta_1, dv_hbeta_1], wl, z, hbeta)
    model_gauss_hbeta_2 = Gauss([a_hbeta_2, sigma_hbeta_2, dv_hbeta_2], wl, z, hbeta)

    linelist = [model_gauss_oiii_a1, model_gauss_oiii_b1, model_gauss_hbeta_1, model_gauss_hbeta_2]
    model_lines = np.sum(np.array(linelist), axis=0)
    
    model = model_pl + model_iron + model_lines
    if components:
        return model_pl, linelist, model_iron
    return model


def flux_model_no_OIII(wl, theta, z, iron, components=False):

    a0, alpha, a_hbeta_1, sigma_hbeta_1, dv_hbeta_1, a_hbeta_2, sigma_hbeta_2, dv_hbeta_2, a_Fe, sigma_Fe, dv_Fe = theta

    model_pl = PowerLaw(a0, alpha, wl, z)

    # add iron template
    fl_iron = np.zeros_like(wl)
    dlam_iron = np.diff(iron[:, 0])[0] * (1.+z)    
    std_gauss = np.sqrt((sigma_Fe*2*np.sqrt(2*np.log(2)))**2 - 860**2)/(2*np.sqrt(2*np.log(2)))
    dlam_hbeta = std_gauss / const.c.to(u.km/u.s).value * hbeta * (1.+z)
    std_gauss_pixels = dlam_hbeta / dlam_iron
    gauss = Gaussian1DKernel(stddev = std_gauss_pixels)
    iron_smoothed = convolve(iron[:, 1], gauss)
    dz_Fe = dv_Fe / const.c.to(u.km/u.s).value * (1+z)
    iron_tem = interpol.interp1d(iron[:, 0] * (1. + z + dz_Fe), iron_smoothed, kind = 'quadratic') # given in 1e-15 ergs cm^-2 s^-1 A^-1
    mask_iron = (wl > iron[0, 0]*(1+z+dz_Fe)) * (wl < iron[-1, 0]*(1+z+dz_Fe))
    fl_iron[mask_iron] = iron_tem(wl[mask_iron])

    model_iron = a_Fe * fl_iron

    model_gauss_hbeta_1 = Gauss([a_hbeta_1, sigma_hbeta_1, dv_hbeta_1], wl, z, hbeta)
    model_gauss_hbeta_2 = Gauss([a_hbeta_2, sigma_hbeta_2, dv_hbeta_2], wl, z, hbeta)

    linelist = [model_gauss_hbeta_1, model_gauss_hbeta_2]
    model_lines = np.sum(np.array(linelist), axis=0)
    
    model = model_pl + model_iron + model_lines
    if components:
        return model_pl, linelist, model_iron
    return model


def flux_model_full(wl, theta, z, iron, components=False):

    a0, alpha, a_oiii_a1, sigma_oiii_a1, dv_oiii_a1, a_hbeta_1, sigma_hbeta_1, dv_hbeta_1, a_hbeta_2, sigma_hbeta_2, dv_hbeta_2, a_Fe, sigma_Fe, dv_Fe = theta

    model_pl = PowerLaw(a0, alpha, wl, z)

    # add iron template
    fl_iron = np.zeros_like(wl)
    #print(iron)
    dlam_iron = np.diff(iron[:, 0])[0] * (1.+z)    
    std_gauss = np.sqrt((sigma_Fe*2*np.sqrt(2*np.log(2)))**2 - 860**2)/(2*np.sqrt(2*np.log(2)))
    dlam_hbeta = std_gauss / const.c.to(u.km/u.s).value * hbeta * (1.+z)
    std_gauss_pixels = dlam_hbeta / dlam_iron
    gauss = Gaussian1DKernel(stddev = std_gauss_pixels)
    iron_smoothed = convolve(iron[:, 1], gauss)
    dz_Fe = dv_Fe / const.c.to(u.km/u.s).value * (1+z)
    iron_tem = interpol.interp1d(iron[:, 0] * (1. + z + dz_Fe), iron_smoothed, kind = 'quadratic') # given in 1e-15 ergs cm^-2 s^-1 A^-1
    mask_iron = (wl > iron[0, 0]*(1+z+dz_Fe)) * (wl < iron[-1, 0]*(1+z+dz_Fe))
    fl_iron[mask_iron] = iron_tem(wl[mask_iron])

    model_iron = a_Fe * fl_iron

    model_gauss_oiii_a1 = Gauss([a_oiii_a1, sigma_oiii_a1, dv_oiii_a1], wl, z, oiii_a)
    model_gauss_oiii_b1 = Gauss([3*a_oiii_a1, sigma_oiii_a1, dv_oiii_a1], wl, z, oiii_b)
    model_gauss_hbeta_1 = Gauss([a_hbeta_1, sigma_hbeta_1, dv_hbeta_1], wl, z, hbeta)
    model_gauss_hbeta_2 = Gauss([a_hbeta_2, sigma_hbeta_2, dv_hbeta_2], wl, z, hbeta)

    linelist = [model_gauss_oiii_a1, model_gauss_oiii_b1, model_gauss_hbeta_1, model_gauss_hbeta_2]
    model_lines = np.sum(np.array(linelist), axis=0)
    
    model = model_pl + model_iron + model_lines
    if components:
        return model_pl, linelist, model_iron
    return model


def flux_model_full_dpl(wl, theta, z, iron, components=False):

    a0, alpha0, a1, alpha1, a_oiii_a1, sigma_oiii_a1, dv_oiii_a1, a_hbeta_1, sigma_hbeta_1, dv_hbeta_1, a_hbeta_2, sigma_hbeta_2, dv_hbeta_2, a_Fe, sigma_Fe, dv_Fe = theta

    model_pl0 = PowerLaw(a0, alpha0, wl, z)
    model_pl1 = PowerLaw(a1, alpha1, wl, z)

    model_pl = model_pl0 + model_pl1

    # add iron template
    fl_iron = np.zeros_like(wl)
    #print(iron)
    dlam_iron = np.diff(iron[:, 0])[0] * (1.+z)    
    std_gauss = np.sqrt((sigma_Fe*2*np.sqrt(2*np.log(2)))**2 - 860**2)/(2*np.sqrt(2*np.log(2)))
    dlam_hbeta = std_gauss / const.c.to(u.km/u.s).value * hbeta * (1.+z)
    std_gauss_pixels = dlam_hbeta / dlam_iron
    gauss = Gaussian1DKernel(stddev = std_gauss_pixels)
    iron_smoothed = convolve(iron[:, 1], gauss)
    dz_Fe = dv_Fe / const.c.to(u.km/u.s).value * (1+z)
    iron_tem = interpol.interp1d(iron[:, 0] * (1. + z + dz_Fe), iron_smoothed, kind = 'quadratic') # given in 1e-15 ergs cm^-2 s^-1 A^-1
    mask_iron = (wl > iron[0, 0]*(1+z+dz_Fe)) * (wl < iron[-1, 0]*(1+z+dz_Fe))
    fl_iron[mask_iron] = iron_tem(wl[mask_iron])

    model_iron = a_Fe * fl_iron

    model_gauss_oiii_a1 = Gauss([a_oiii_a1, sigma_oiii_a1, dv_oiii_a1], wl, z, oiii_a)
    model_gauss_oiii_b1 = Gauss([3*a_oiii_a1, sigma_oiii_a1, dv_oiii_a1], wl, z, oiii_b)
    model_gauss_hbeta_1 = Gauss([a_hbeta_1, sigma_hbeta_1, dv_hbeta_1], wl, z, hbeta)
    model_gauss_hbeta_2 = Gauss([a_hbeta_2, sigma_hbeta_2, dv_hbeta_2], wl, z, hbeta)

    linelist = [model_gauss_oiii_a1, model_gauss_oiii_b1, model_gauss_hbeta_1, model_gauss_hbeta_2]
    model_lines = np.sum(np.array(linelist), axis=0)
    
    model = model_pl + model_iron + model_lines
    if components:
        return model_pl, linelist, model_iron
    return model


def flux_model_full_with_Hgamma(wl, theta, z, iron, components=False):

    a0, alpha, a_oiii_a1, sigma_oiii_a1, dv_oiii_a1, a_hbeta_1, sigma_hbeta_1, dv_hbeta_1, a_hbeta_2, sigma_hbeta_2, dv_hbeta_2,\
            a_gamma_1, sigma_gamma_1, dv_gamma_1, a_gamma_2, sigma_gamma_2, dv_gamma_2, a_Fe, sigma_Fe, dv_Fe = theta

    model_pl = PowerLaw(a0, alpha, wl, z)

    # add iron template
    fl_iron = np.zeros_like(wl)
    #print(iron)
    dlam_iron = np.diff(iron[:, 0])[0] * (1.+z)    
    std_gauss = np.sqrt((sigma_Fe*2*np.sqrt(2*np.log(2)))**2 - 860**2)/(2*np.sqrt(2*np.log(2)))
    dlam_hbeta = std_gauss / const.c.to(u.km/u.s).value * hbeta * (1.+z)
    std_gauss_pixels = dlam_hbeta / dlam_iron
    gauss = Gaussian1DKernel(stddev = std_gauss_pixels)
    iron_smoothed = convolve(iron[:, 1], gauss)
    dz_Fe = dv_Fe / const.c.to(u.km/u.s).value * (1+z)
    iron_tem = interpol.interp1d(iron[:, 0] * (1. + z + dz_Fe), iron_smoothed, kind = 'quadratic') # given in 1e-15 ergs cm^-2 s^-1 A^-1
    mask_iron = (wl > iron[0, 0]*(1+z+dz_Fe)) * (wl < iron[-1, 0]*(1+z+dz_Fe))
    fl_iron[mask_iron] = iron_tem(wl[mask_iron])

    model_iron = a_Fe * fl_iron

    model_gauss_oiii_a1 = Gauss([a_oiii_a1, sigma_oiii_a1, dv_oiii_a1], wl, z, oiii_a)
    model_gauss_oiii_b1 = Gauss([3*a_oiii_a1, sigma_oiii_a1, dv_oiii_a1], wl, z, oiii_b)
    model_gauss_hbeta_1 = Gauss([a_hbeta_1, sigma_hbeta_1, dv_hbeta_1], wl, z, hbeta)
    model_gauss_hbeta_2 = Gauss([a_hbeta_2, sigma_hbeta_2, dv_hbeta_2], wl, z, hbeta)

    model_gauss_hgamma_1 = Gauss([a_gamma_1, sigma_gamma_1, dv_gamma_1], wl, z, hgamma)
    model_gauss_hgamma_2 = Gauss([a_gamma_2, sigma_gamma_2, dv_gamma_2], wl, z, hgamma)

    linelist = [model_gauss_oiii_a1, model_gauss_oiii_b1, model_gauss_hbeta_1, model_gauss_hbeta_2, model_gauss_hgamma_1, model_gauss_hgamma_2]
    model_lines = np.sum(np.array(linelist), axis=0)
    
    model = model_pl + model_iron + model_lines
    if components:
        return model_pl, linelist, model_iron
    return model


def flux_model_full_with_Hgamma_twocomp(wl, theta, z, iron, components=False):

    a0, alpha,\
    a_oiii_a1, sigma_oiii_a1, dv_oiii_a1, a_oiii_a2, sigma_oiii_a2, dv_oiii_a2,\
    a_hbeta_1, sigma_hbeta_1, dv_hbeta_1, a_hbeta_2, sigma_hbeta_2, dv_hbeta_2,\
    a_gamma_1, sigma_gamma_1, dv_gamma_1, a_gamma_2, sigma_gamma_2, dv_gamma_2,\
    a_Fe_1, sigma_Fe_1, dv_Fe_1,a_Fe_2, sigma_Fe_2, dv_Fe_2 = theta

    model_pl = PowerLaw(a0, alpha, wl, z)

    model_iron_1 = ironcolvolve([a_Fe_1, sigma_Fe_1, dv_Fe_1], wl, z, iron)
    model_iron_2 = ironcolvolve([a_Fe_2, sigma_Fe_2, dv_Fe_2], wl, z, iron)
    model_iron = model_iron_1 + model_iron_2


    model_gauss_oiii_a1 = Gauss([a_oiii_a1, sigma_oiii_a1, dv_oiii_a1], wl, z, oiii_a)
    model_gauss_oiii_b1 = Gauss([3*a_oiii_a1, sigma_oiii_a1, dv_oiii_a1], wl, z, oiii_b)
    model_gauss_oiii_a2 = Gauss([a_oiii_a2, sigma_oiii_a2, dv_oiii_a2], wl, z, oiii_a)
    model_gauss_oiii_b2 = Gauss([3*a_oiii_a2, sigma_oiii_a2, dv_oiii_a2], wl, z, oiii_b)

    model_gauss_hbeta_1 = Gauss([a_hbeta_1, sigma_hbeta_1, dv_hbeta_1], wl, z, hbeta)
    model_gauss_hbeta_2 = Gauss([a_hbeta_2, sigma_hbeta_2, dv_hbeta_2], wl, z, hbeta)

    model_gauss_hgamma_1 = Gauss([a_gamma_1, sigma_gamma_1, dv_gamma_1], wl, z, hgamma)
    model_gauss_hgamma_2 = Gauss([a_gamma_2, sigma_gamma_2, dv_gamma_2], wl, z, hgamma)

    linelist = [model_gauss_oiii_a1, model_gauss_oiii_b1, model_gauss_oiii_a2, model_gauss_oiii_b2,\
                model_gauss_hbeta_1, model_gauss_hbeta_2, model_gauss_hgamma_1, model_gauss_hgamma_2]
    model_lines = np.sum(np.array(linelist), axis=0)
    
    model = model_pl + model_iron + model_lines
    if components:
        return model_pl, linelist, model_iron
    return model




def flux_model_full_with_Hgamma_freeOIII(wl, theta, z, iron, components=False):

    a0, alpha, a_oiii_a1, sigma_oiii_a1, dv_oiii_a1, a_oiii_b1, sigma_oiii_b1, dv_oiii_b1, a_hbeta_1, sigma_hbeta_1, dv_hbeta_1, a_hbeta_2, sigma_hbeta_2, dv_hbeta_2,\
            a_gamma_1, sigma_gamma_1, dv_gamma_1, a_gamma_2, sigma_gamma_2, dv_gamma_2, a_Fe, sigma_Fe, dv_Fe = theta

    model_pl = PowerLaw(a0, alpha, wl, z)

    # add iron template
    fl_iron = np.zeros_like(wl)
    #print(iron)
    dlam_iron = np.diff(iron[:, 0])[0] * (1.+z)    
    std_gauss = np.sqrt((sigma_Fe*2*np.sqrt(2*np.log(2)))**2 - 860**2)/(2*np.sqrt(2*np.log(2)))
    dlam_hbeta = std_gauss / const.c.to(u.km/u.s).value * hbeta * (1.+z)
    std_gauss_pixels = dlam_hbeta / dlam_iron
    gauss = Gaussian1DKernel(stddev = std_gauss_pixels)
    iron_smoothed = convolve(iron[:, 1], gauss)
    dz_Fe = dv_Fe / const.c.to(u.km/u.s).value * (1+z)
    iron_tem = interpol.interp1d(iron[:, 0] * (1. + z + dz_Fe), iron_smoothed, kind = 'quadratic') # given in 1e-15 ergs cm^-2 s^-1 A^-1
    mask_iron = (wl > iron[0, 0]*(1+z+dz_Fe)) * (wl < iron[-1, 0]*(1+z+dz_Fe))
    fl_iron[mask_iron] = iron_tem(wl[mask_iron])

    model_iron = a_Fe * fl_iron

    model_gauss_oiii_a1 = Gauss([a_oiii_a1, sigma_oiii_a1, dv_oiii_a1], wl, z, oiii_a)
    model_gauss_oiii_b1 = Gauss([a_oiii_b1, sigma_oiii_b1, dv_oiii_b1], wl, z, oiii_b)
    model_gauss_hbeta_1 = Gauss([a_hbeta_1, sigma_hbeta_1, dv_hbeta_1], wl, z, hbeta)
    model_gauss_hbeta_2 = Gauss([a_hbeta_2, sigma_hbeta_2, dv_hbeta_2], wl, z, hbeta)

    model_gauss_hgamma_1 = Gauss([a_gamma_1, sigma_gamma_1, dv_gamma_1], wl, z, hgamma)
    model_gauss_hgamma_2 = Gauss([a_gamma_2, sigma_gamma_2, dv_gamma_2], wl, z, hgamma)

    linelist = [model_gauss_oiii_a1, model_gauss_oiii_b1, model_gauss_hbeta_1, model_gauss_hbeta_2, model_gauss_hgamma_1, model_gauss_hgamma_2]
    model_lines = np.sum(np.array(linelist), axis=0)
    
    model = model_pl + model_iron + model_lines
    if components:
        return model_pl, linelist, model_iron
    return model


def flux_model_full_free_OIII(wl, theta, z, iron, components=False):

    a0, alpha, a_oiii_a1, sigma_oiii_a1, dv_oiii_a1, a_oiii_b1, sigma_oiii_b1, dv_oiii_b1,\
    a_hbeta_1, sigma_hbeta_1, dv_hbeta_1, a_hbeta_2, sigma_hbeta_2, dv_hbeta_2, a_Fe, sigma_Fe, dv_Fe = theta

    model_pl = PowerLaw(a0, alpha, wl, z)

    # add iron template
    fl_iron = np.zeros_like(wl)
    dlam_iron = np.diff(iron[:, 0])[0] * (1.+z)    
    std_gauss = np.sqrt((sigma_Fe*2*np.sqrt(2*np.log(2)))**2 - 860**2)/(2*np.sqrt(2*np.log(2)))
    dlam_hbeta = std_gauss / const.c.to(u.km/u.s).value * hbeta * (1.+z)
    std_gauss_pixels = dlam_hbeta / dlam_iron
    gauss = Gaussian1DKernel(stddev = std_gauss_pixels)
    iron_smoothed = convolve(iron[:, 1], gauss)
    dz_Fe = dv_Fe / const.c.to(u.km/u.s).value * (1+z)
    iron_tem = interpol.interp1d(iron[:, 0] * (1. + z + dz_Fe), iron_smoothed, kind = 'quadratic') # given in 1e-15 ergs cm^-2 s^-1 A^-1
    mask_iron = (wl > iron[0, 0]*(1+z+dz_Fe)) * (wl < iron[-1, 0]*(1+z+dz_Fe))
    fl_iron[mask_iron] = iron_tem(wl[mask_iron])

    model_iron = a_Fe * fl_iron

    model_gauss_oiii_a1 = Gauss([a_oiii_a1, sigma_oiii_a1, dv_oiii_a1], wl, z, oiii_a)
    model_gauss_oiii_b1 = Gauss([a_oiii_b1, sigma_oiii_b1, dv_oiii_b1], wl, z, oiii_b)
    model_gauss_hbeta_1 = Gauss([a_hbeta_1, sigma_hbeta_1, dv_hbeta_1], wl, z, hbeta)
    model_gauss_hbeta_2 = Gauss([a_hbeta_2, sigma_hbeta_2, dv_hbeta_2], wl, z, hbeta)

    linelist = [model_gauss_oiii_a1, model_gauss_oiii_b1, model_gauss_hbeta_1, model_gauss_hbeta_2]
    model_lines = np.sum(np.array(linelist), axis=0)
    
    model = model_pl + model_iron + model_lines
    if components:
        return model_pl, linelist, model_iron
    return model

def flux_model_full_two_OIII(wl, theta, z, iron, components=False):

    a0, alpha, a_oiii_a1, sigma_oiii_a1, dv_oiii_a1, a_oiii_a2, sigma_oiii_a2, dv_oiii_a2,\
    a_hbeta_1, sigma_hbeta_1, dv_hbeta_1, a_hbeta_2, sigma_hbeta_2, dv_hbeta_2, a_Fe, sigma_Fe, dv_Fe = theta

    model_pl = PowerLaw(a0, alpha, wl, z)

    # add iron template
    fl_iron = np.zeros_like(wl)
    dlam_iron = np.diff(iron[:, 0])[0] * (1.+z)    
    std_gauss = np.sqrt((sigma_Fe*2*np.sqrt(2*np.log(2)))**2 - 860**2)/(2*np.sqrt(2*np.log(2)))
    dlam_hbeta = std_gauss / const.c.to(u.km/u.s).value * hbeta * (1.+z)
    std_gauss_pixels = dlam_hbeta / dlam_iron
    gauss = Gaussian1DKernel(stddev = std_gauss_pixels)
    iron_smoothed = convolve(iron[:, 1], gauss)
    dz_Fe = dv_Fe / const.c.to(u.km/u.s).value * (1+z)
    iron_tem = interpol.interp1d(iron[:, 0] * (1. + z + dz_Fe), iron_smoothed, kind = 'quadratic') # given in 1e-15 ergs cm^-2 s^-1 A^-1
    mask_iron = (wl > iron[0, 0]*(1+z+dz_Fe)) * (wl < iron[-1, 0]*(1+z+dz_Fe))
    fl_iron[mask_iron] = iron_tem(wl[mask_iron])

    model_iron = a_Fe * fl_iron

    model_gauss_oiii_a1 = Gauss([a_oiii_a1, sigma_oiii_a1, dv_oiii_a1], wl, z, oiii_a)
    model_gauss_oiii_b1 = Gauss([3*a_oiii_a1, sigma_oiii_a1, dv_oiii_a1], wl, z, oiii_b)
    model_gauss_oiii_a2 = Gauss([a_oiii_a2, sigma_oiii_a2, dv_oiii_a2], wl, z, oiii_a)
    model_gauss_oiii_b2 = Gauss([3*a_oiii_a2, sigma_oiii_a2, dv_oiii_a2], wl, z, oiii_b)
    model_gauss_hbeta_1 = Gauss([a_hbeta_1, sigma_hbeta_1, dv_hbeta_1], wl, z, hbeta)
    model_gauss_hbeta_2 = Gauss([a_hbeta_2, sigma_hbeta_2, dv_hbeta_2], wl, z, hbeta)

    linelist = [model_gauss_oiii_a1, model_gauss_oiii_b1,\
                model_gauss_oiii_a2, model_gauss_oiii_b2,\
                model_gauss_hbeta_1, model_gauss_hbeta_2]
    model_lines = np.sum(np.array(linelist), axis=0)
    
    model = model_pl + model_iron + model_lines
    if components:
        return model_pl, linelist, model_iron
    return model


def flux_model_full_two_OIII_two_iron(wl, theta, z, iron, components=False):

    a0, alpha, a_oiii_a1, sigma_oiii_a1, dv_oiii_a1, a_oiii_a2, sigma_oiii_a2, dv_oiii_a2,\
    a_hbeta_1, sigma_hbeta_1, dv_hbeta_1, a_hbeta_2, sigma_hbeta_2, dv_hbeta_2,\
    a_Fe_1, sigma_Fe_1, dv_Fe_1, a_Fe_2, sigma_Fe_2, dv_Fe_2  = theta

    model_pl = PowerLaw(a0, alpha, wl, z)

    # add iron template
    model_iron_1 = ironcolvolve([a_Fe_1, sigma_Fe_1, dv_Fe_1], wl, z, iron)
    model_iron_2 = ironcolvolve([a_Fe_2, sigma_Fe_2, dv_Fe_2], wl, z, iron)
    model_iron = model_iron_1 + model_iron_2

    model_gauss_oiii_a1 = Gauss([a_oiii_a1, sigma_oiii_a1, dv_oiii_a1], wl, z, oiii_a)
    model_gauss_oiii_b1 = Gauss([3*a_oiii_a1, sigma_oiii_a1, dv_oiii_a1], wl, z, oiii_b)
    model_gauss_oiii_a2 = Gauss([a_oiii_a2, sigma_oiii_a2, dv_oiii_a2], wl, z, oiii_a)
    model_gauss_oiii_b2 = Gauss([3*a_oiii_a2, sigma_oiii_a2, dv_oiii_a2], wl, z, oiii_b)
    model_gauss_hbeta_1 = Gauss([a_hbeta_1, sigma_hbeta_1, dv_hbeta_1], wl, z, hbeta)
    model_gauss_hbeta_2 = Gauss([a_hbeta_2, sigma_hbeta_2, dv_hbeta_2], wl, z, hbeta)

    linelist = [model_gauss_oiii_a1, model_gauss_oiii_b1,\
                model_gauss_oiii_a2, model_gauss_oiii_b2,\
                model_gauss_hbeta_1, model_gauss_hbeta_2]
    model_lines = np.sum(np.array(linelist), axis=0)
    
    model = model_pl + model_iron + model_lines
    if components:
        return model_pl, linelist, model_iron
    return model



def lnlike_jwst(wl, flux, ivar, theta, z, iron, modelfunc):
 
    model = modelfunc(wl, theta, z, iron)
    lnlike = -0.5 * (flux - model)**2 * ivar
    return np.sum(lnlike)

def Gauss(theta, wl, z, line):
    A, sigma, dv = theta
    sigma_wl = sigma / const.c.to(u.km/u.s).value * line * (1.+z)
    mean = line * (1.+z) + (dv/(const.c.to(u.km/u.s).value)) * line * (1.+z)
    return A * np.exp(-0.5 * (wl - mean)**2 / sigma_wl**2)

def ironcolvolve(theta, wl, z, iron):
    A, sigma, dv = theta

    fl_iron = np.zeros_like(wl)
    dlam_iron = np.diff(iron[:, 0])[0] * (1.+z)    
    std_gauss = np.sqrt((sigma*2*np.sqrt(2*np.log(2)))**2 - 860**2)/(2*np.sqrt(2*np.log(2)))
    dlam_hbeta = std_gauss / const.c.to(u.km/u.s).value * hbeta * (1.+z)
    std_gauss_pixels = dlam_hbeta / dlam_iron
    gauss = Gaussian1DKernel(stddev = std_gauss_pixels)
    iron_smoothed = convolve(iron[:, 1], gauss)
    dz_Fe = dv / const.c.to(u.km/u.s).value * (1+z)
    iron_tem = interpol.interp1d(iron[:, 0] * (1. + z + dz_Fe), iron_smoothed, kind = 'quadratic') # given in 1e-15 ergs cm^-2 s^-1 A^-1
    mask_iron = (wl > iron[0, 0]*(1+z+dz_Fe)) * (wl < iron[-1, 0]*(1+z+dz_Fe))
    fl_iron[mask_iron] = iron_tem(wl[mask_iron])
    model_iron = A * fl_iron

    return model_iron

def PowerLaw(F0, alpha, wl, z):    
    return F0 * (wl/(5000.*(1.+z)))**alpha

def readdata(filename, wlmin, wlmax):
    stack1d = fits.open(filename)[1].data

    wl = stack1d['wavelength']
    flux = stack1d['flux']
    ivar = stack1d['ivar']

    mask = (wl>wlmin)&(wl<wlmax)

    wl = wl[mask]
    flux = flux[mask]
    ivar = ivar[mask]

    return wl, flux, ivar

