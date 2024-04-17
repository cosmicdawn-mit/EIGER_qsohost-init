import os, sys
import numpy as np
import matplotlib.pyplot as plt

import pickle
from astropy.io import fits
from astropy.table import Table

#import corner

import fit_mcmc_bases as bases

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "Computer Modern Serif",
})

def plot_fit_oneax(ax, wl, flux, theta, z, modelfunc, iron, wave_range,\
                   ylim, title='', addlegend=True):

    modelflux = modelfunc(wl, theta, z, iron)
    cont, lines, iron = modelfunc(wl, theta, z, iron, components=True)
    lines = np.array(lines)

    ax.step(wl, flux, 'k', where='mid')
    ax.plot(wl, modelflux, color='r', label='Model')
    ax.plot(wl, cont, color='gray', label='Power Law')

    if len(np.array([iron]).flatten())>1:
        ax.plot(wl, iron, '--', label='FeII', color='darkorange')


    # OIII beta components
    ax.plot(wl, lines[0], 'g--', label='[O III]')
    ax.plot(wl, lines[1], 'g--')
    ax.plot(wl, lines[2], 'g--')
    ax.plot(wl, lines[3], 'g--')

    ax.plot(wl, lines[4], 'b--', label=r'H$\beta$')
    ax.plot(wl, lines[5], 'b--')

    # mark the lines

    line_wls = [4861.4, 4958.9, 5006.8]
    line_pos = [4861.4-30, 4970-30]
    line_labels = [r'H$\beta$', r'[O III]']

    ax.plot(np.array(line_wls)*(1+z), [np.max(modelflux)*1.05]*len(line_wls), 'k|', ms=10)

    for index in range(len(line_pos)):
        ax.text(line_pos[index]*(1+z), np.max(modelflux)*1.1, line_labels[index], fontsize=12)

    ax.tick_params(axis="both", direction="in")
    ax.tick_params(axis="both", direction="in", which='minor')

#    ax.set_xlabel(r'Wavelength $\mathrm{[\AA]}$', fontsize=12)
#    ax.set_ylabel(r'Flux $\mathrm{[10^{-18}erg~s^{-1}cm^{-2}{\AA}^{-1}]}$',fontsize=12)

    ax.set_xlim(wave_range)
    ax.set_ylim(ylim)

    if addlegend:
        ax.legend(fontsize=12, frameon=False)
    ax.set_title(title, fontsize=14)

    return ax

def plot_fit_J1120ax(ax, wl, flux, theta, z, modelfunc, iron, wave_range,\
                   ylim, title='', addlegend=True):

    modelflux = modelfunc(wl, theta, z, iron)
    cont, lines, iron = modelfunc(wl, theta, z, iron, components=True)
    lines = np.array(lines)

    ax.step(wl, flux, 'k', where='mid')
    ax.plot(wl, modelflux, color='r')
    ax.plot(wl, cont, color='gray')

    if len(np.array([iron]).flatten())>1:
        ax.plot(wl, iron, '--', color='darkorange')


    # OIII beta components
    ax.plot(wl, lines[6], 'm--', label=r'H$\gamma$')
    ax.plot(wl, lines[7], 'm--')


    ax.plot(wl, lines[0], 'g--')
    ax.plot(wl, lines[1], 'g--')
    ax.plot(wl, lines[2], 'g--')
    ax.plot(wl, lines[3], 'g--')

    ax.plot(wl, lines[4], 'b--')
    ax.plot(wl, lines[5], 'b--')


    # mark the lines

    line_wls = [4861.4, 4958.9, 5006.8, 4340.472]
    line_pos = [4861.4-30, 4960-30, 4340-30]
    line_labels = [r'H$\beta$', r'[O III]', r'H$\gamma$']

    ax.plot(np.array(line_wls)*(1+z), [np.max(modelflux)*1.05]*len(line_wls), 'k|', ms=10)

    for index in range(len(line_pos)):
        ax.text(line_pos[index]*(1+z), np.max(modelflux)*1.1, line_labels[index], fontsize=12)

    ax.tick_params(axis="both", direction="in")
    ax.tick_params(axis="both", direction="in", which='minor')

#    ax.set_xlabel(r'Wavelength $\mathrm{[\AA]}$', fontsize=12)
#    ax.set_ylabel(r'Flux $\mathrm{[10^{-18}erg~s^{-1}cm^{-2}{\AA}^{-1}]}$',fontsize=12)

    ax.set_xlim(wave_range)
    ax.set_ylim(ylim)

    if addlegend:
        ax.legend(fontsize=12, frameon=False, loc='upper center')
    ax.set_title(title, fontsize=14)

    return ax



def plot_all_quasars():
    iron = Table.read('../../data/irontemp_park22.txt', format='mrt')
    iron = np.array([iron['Wave'], iron['Flux'], iron['e_Flux']]).T

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=[10,10])

    # J0100
    mcmcfile_J0100 = '../saves/J0100_mcmc_samples_step2_final.pickle'
    specfile_J0100 = '../../data/J0100_stack1dnew.fits'
    z_J0100 = 6.327
    modelfunc_J0100 = bases.flux_model_full_two_OIII_two_iron

    mcmc_list = pickle.load(open(mcmcfile_J0100, 'rb'))
    theta_med = np.median(mcmc_list, axis=0)

    wl, flux, ivar = bases.readdata(specfile_J0100, wlmin=32000, wlmax=40000)

    plot_fit_oneax(axes[0][0], wl, flux, theta_med, z_J0100, modelfunc_J0100, iron, [33000, 39500],\
                   ylim=[0, 16.0], title='J0100+2802')


    # J0148+0600
    mcmcfile_J0148 = '../saves/J0148_mcmc_samples_step2_twoOIII_two_iron.pickle'
    specfile_J0148 = '../test/J0148_stack1d.fits'
    z_J0148 = 5.98
    modelfunc_J0148 = bases.flux_model_full_two_OIII_two_iron

    mcmc_list = pickle.load(open(mcmcfile_J0148, 'rb'))
    theta_med = np.median(mcmc_list, axis=0)

    wl, flux, ivar = bases.readdata(specfile_J0148, wlmin=32000, wlmax=40000)

    plot_fit_oneax(axes[0][1], wl, flux, theta_med, z_J0148, modelfunc_J0148, iron, [32000, 38500],\
                   ylim=[-0, 4.0], title='J0148+0600', addlegend=False)


    # J1030
    mcmcfile_J1030 = '../saves/J1030_mcmc_samples_step2_final.pickle'
    specfile_J1030 = '../../data/J1030_stack1dnew.fits'
    z_J1030 = 6.304
    modelfunc_J1030 = bases.flux_model_full_two_OIII_two_iron

    mcmc_list = pickle.load(open(mcmcfile_J1030, 'rb'))
    theta_med = np.median(mcmc_list, axis=0)

    print(theta_med)

    wl, flux, ivar = bases.readdata(specfile_J1030, wlmin=32000, wlmax=40000)

    plot_fit_oneax(axes[1][0], wl, flux, theta_med, z_J1030, modelfunc_J1030, iron, [33000, 39500],\
                   ylim=[-0, 3.5], title='J1030+0524', addlegend=False)

    # J159
    mcmcfile_J159 = '../saves/J159_mcmc_samples_step2_final.pickle'
    specfile_J159 = '../../data/J159_stack1dnew.fits'
    z_J159 = 6.381
    modelfunc_J159 = bases.flux_model_full_two_OIII_two_iron

    mcmc_list = pickle.load(open(mcmcfile_J159, 'rb'))
    theta_med = np.median(mcmc_list, axis=0)

    wl, flux, ivar = bases.readdata(specfile_J159, wlmin=32000, wlmax=40000)

    plot_fit_oneax(axes[1][1], wl, flux, theta_med, z_J159, modelfunc_J159, iron, [33000, 39500],\
                   ylim=[-0, 2.8], title='J159-02', addlegend=False)


    # J1120+0641
    mcmcfile_J1120 = '../saves/J1120_mcmc_samples_step2_final.pickle'
    specfile_J1120 = '../../data/J1120_stack1d.fits'

    z_J1120 = 7.08
    modelfunc_J1120 = bases.flux_model_full_with_Hgamma_twocomp

    mcmc_list = pickle.load(open(mcmcfile_J1120, 'rb'))
    theta_med = np.median(mcmc_list, axis=0)

    wl, flux, ivar = bases.readdata(specfile_J1120, wlmin=34000, wlmax=40500)

    plot_fit_J1120ax(axes[2][0], wl, flux, theta_med, z_J1120, modelfunc_J1120, iron, [34000, 40500],\
                   ylim=[-0, 2.0], title='J1120+0641', addlegend=True)


    # J1148+5251
    mcmcfile_J1148 = '../saves/J1148_mcmc_samples_step2_final.pickle'
    specfile_J1148 = '../../data/J1148_stack1d.fits'

    z_J1148 = 6.42
    modelfunc_J1148 = bases.flux_model_full_two_OIII_two_iron

    mcmc_list = pickle.load(open(mcmcfile_J1148, 'rb'))
    theta_med = np.median(mcmc_list, axis=0)

    wl, flux, ivar = bases.readdata(specfile_J1148, wlmin=33000, wlmax=39500)

    plot_fit_oneax(axes[2][1], wl, flux, theta_med, z_J1148, modelfunc_J1148, iron, [33000, 39500],\
                   ylim=[-0, 4.3], title='J1148+5251', addlegend=False)


    fig.supxlabel('Observed Wavelength [Angstrom]', fontsize=16)
    fig.supylabel(r'Flux $\mathrm{[10^{-18}erg~s^{-1}cm^{-2}{\AA}^{-1}]}$',fontsize=16)

    plt.tight_layout()

#    plt.savefig('./allqso_1dspec.pdf')
    plt.show()


if __name__=='__main__':
    plot_all_quasars()
