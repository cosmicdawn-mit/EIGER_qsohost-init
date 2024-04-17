import os, sys
import numpy as np
import emcee

import matplotlib.pyplot as plt

import pickle
import fit_mcmc_bases as bases

from astropy.table import Table


#def fit_J1120_full():



def main():
    z = 6.381
    qname_short = 'J159'

    wl, flux, ivar = bases.readdata('../../data/%s_stack1dnew.fits'%qname_short, wlmin=33000, wlmax=39500)

    # first round
    a0guess = 1.2
    alphaguess = -1

    coef_min = np.array([0, -5,# power law
                         0, 10, -5000,# OIIIa
                         0, 10, -5000,# OIIIb
                         0, 10, -5000,# Hba
                         0, 1500, -5000,#Hbb
                        0, 2000, -5000,
                        0, 860, -5000])#iron

    coef_max = np.array([10, 2,\
                         1, 1000, 5000,\
                         1, 4000, 5000,\
                         5, 1500, 5000,
                         5, 4000, 5000,
                        5, 4000, 5000,
                        5, 2000, 5000])


    ndim = len(coef_min)
    nwalkers = 5 * ndim

    pos = bases.GetInitialPositions(nwalkers, coef_min, coef_max, ndim)

    from_previous = True
    savefile = '../saves/%s_mcmc_samples_step1_twocomp.pickle'%qname_short

    if from_previous:
        samples = pickle.load(open('../saves/%s_mcmc_samples_step2_final.pickle'%qname_short, 'rb'))
        medsample = np.median(samples, axis=0)
        best_guess = medsample
        best_guess[3] = 500
        best_guess[4] = 0
        print(best_guess)

        for index in range(len(best_guess)):
            print(coef_min[index], best_guess[index], coef_max[index])
            print((coef_min[index]<best_guess[index]) and (best_guess[index]<coef_max[index]))

        savefile = '../saves/%s_mcmc_samples_step2_final.pickle'%qname_short

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

    f = open(savefile, 'wb')

    pickle.dump(sampler.chain[:,nburn:,:].reshape((-1,ndim)), f)
    f.close()

if __name__=='__main__':
    main()

