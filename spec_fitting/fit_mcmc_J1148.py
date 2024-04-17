import os, sys
import numpy as np
import zeus, emcee

import matplotlib.pyplot as plt

import pickle
import fit_mcmc_bases as bases

from astropy.table import Table


def main():
    # J1120: z=7.08
    z = 6.42

    wl, flux, ivar = bases.readdata('../../data/J1148_stack1d.fits', wlmin=33000, wlmax=39500)

    # first round
    a0guess = 1.2
    alphaguess = -1

    coef_min = np.array([0, -5,# power law
                         0, 10, -350,# OIIIa
                         0, 10, -350,# OIIIb
                         0, 10, -2000,# Hba
                         0, 1500, -2000,#Hbb
                        0, 2000, -2000,
                        0, 860, -2000])#iron

    coef_max = np.array([10, 2,\
                         1, 1000, -250,\
                         1, 4000, -250,\
                         5, 1500, 2000,
                         5, 4000, 2000,
                        5, 6000, 2000,
                        5, 2000, 2000])


    ndim = len(coef_min)
    print(ndim)
    nwalkers = 5 * ndim

#    pos = bases.GetInitialPositions(nwalkers, coef_min, coef_max, ndim)

#    best_guess = np.array([0.75, -1, 0.11, 500, -300, 0.27, 793, 702, 0.48, 2049, -22.08, 0.51, 1056, 431])

    samples = pickle.load(open('../saves/J1148_mcmc_samples_step2_final.pickle', 'rb'))
    medsample = np.median(samples, axis=0)
    best_guess = medsample

    for index in range(len(best_guess)):
        print(coef_min[index], best_guess[index], coef_max[index])
        print((coef_min[index]<best_guess[index]) and (best_guess[index]<coef_max[index]))


    pos = bases.GetInitialPositionsBalls(nwalkers, best_guess, ndim, percent=0.05)

    iron = Table.read('../../data/irontemp_park22.txt', format='mrt')
    iron = np.array([iron['Wave'], iron['Flux'], iron['e_Flux']]).T

    # run mcmc
    nburn, nsample = 20000, 10000
    nsteps = nburn + nsample

    from multiprocessing import Pool

    with Pool() as pool:

        sampler = emcee.EnsembleSampler(nwalkers, ndim, bases.lnprob_jwst, args=(wl, flux, ivar, coef_min, coef_max, z, iron, bases.flux_model_full_two_OIII_two_iron),\
                                        pool=pool, moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)])
        sampler.run_mcmc(pos, nsteps, progress=True)

    f = open('../saves/J1148_mcmc_samples_step2_final.pickle', 'wb')

    pickle.dump(sampler.chain[:,nburn:,:].reshape((-1,ndim)), f)
    f.close()

if __name__=='__main__':
    main()

