import os, sys
import numpy as np
import zeus, emcee

import matplotlib.pyplot as plt

import pickle
import fit_mcmc_bases as bases

from astropy.table import Table

def main():
    # J1120: z=7.08
    z = 5.98

    wl, flux, ivar = bases.readdata('../test/J0148_stack1d.fits', wlmin=32000, wlmax=39500)

    # first round
    a0guess = 2
    alphaguess = -2

    coef_min = np.array([0, -6,# power law,
#                         0, -3,\
                         0, 100, -2000,# OIIIa
                         0, 2500, -2000,# OIIIb
                         0, 100, -2000,# Hba
                         0, 2500, -2000,#Hbb
                         0, 860, -2000,\
                         0, 1500, -2000])#[:-3]#FeII

    coef_max = np.array([10, 1,\
#                         10,2,
                         10, 2500, 2000,\
                         10, 10000, 2000,\
                         10, 2500, 2000,\
                         10, 10000, 5000,
                         10, 2500, 2000,
                         10, 10000, 2000])#[:-3]#FeII


    ndim = len(coef_min)
    nwalkers = 4 * ndim

    iron = Table.read('../../data/irontemp_park22.txt', format='mrt')
    iron = np.array([iron['Wave'], iron['Flux'], iron['e_Flux']]).T

    samples = pickle.load(open('../saves/J0148_mcmc_samples_step2_final.pickle', 'rb'))
    medsample = np.median(samples, axis=0)
    best_guess = medsample
#    best_guess[1] = -3.5
#    best_guess[2] = 0.1
#    best_guess[3] = 1000
#    best_guess[4] = 0
#    best_guess[5] = 0.1
#    best_guess[6] = 3000
#    best_guess[7] = 0
#    best_guess[9] = 3000
#    best_guess[10] = 0
#    best_guess[12] = 3000
#    best_guess[13] = 0

#    best_guess[-3] = 0.7
#    best_guess[-2] = 3000
#    best_guess[-1] = 0

#    best_guess = list(best_guess)
#    best_guess = best_guess + [1, 4000, 0]
#    best_guess = np.array(best_guess)

    for index in range(len(best_guess)):
        print(coef_min[index], best_guess[index], coef_max[index])
#    return 0

    # change it to free OIII
#    best_guess = np.concatenate([medsample[:2], medsample[2:5], medsample[2:5], medsample[5:]])

#    pos = bases.GetInitialPositions(nwalkers, coef_min, coef_max, ndim)

#    best_guess = [1.71, -2.42, 0.06, 1596, 0, 0.57, 1816, -236, 0.74, 3000, 2471, 1.70, 2000, -519]
    pos = bases.GetInitialPositionsBalls(nwalkers, best_guess, ndim, percent=0.05)

    # run mcmc
    nburn, nsample = 10000, 1000
    nsteps = nburn + nsample

    from multiprocessing import Pool

    with Pool() as pool:

        sampler = emcee.EnsembleSampler(nwalkers, ndim, bases.lnprob_jwst, args=(wl, flux, ivar, coef_min, coef_max, z, iron, bases.flux_model_full_two_OIII_two_iron),\
                                        pool=pool, moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)])
        sampler.run_mcmc(pos, nsteps, progress=True)

    f = open('../saves/J0148_mcmc_samples_step2_final.pickle', 'wb')

    pickle.dump(sampler.chain[:,nburn:,:].reshape((-1,ndim)), f)
    f.close()

if __name__=='__main__':
    main()

