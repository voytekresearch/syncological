#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Usage: ei4.py PATH K 
    (--ing | --ping)
    [--no_balanced]
    [--stim_seed=STIM_SEED]
    [--conn_seed=CONN_SEED]
    [--n_stim=NSTIM]
    [--n_job=NJOB]
    [--restart_k=RK]

Simulate K EI circuit, randomizing all weights.

    Arguments:
        PATH        path to save results 
        K           number of simulations to run

    Options:
        -h --help                   show this screen
        --ing                       run in ING mode
        --ping                      run in PING mode
        --no_balanced               turn off balanced background activity     
        --stim_seed=STIM_SEED       seed for creating the stimulus [default: 42]
        --conn_seed=CONN_SEED       seed for creating connections 
        --n_stim=NSTIM              number of driving neurons [default: 100]
        --n_job=NJOB                number of parallel jobs [default: 10]
        --restart_k=RK              'restart' the stimulation at k

"""
from __future__ import division

import numpy as np
import os, sys
import pudb

import syncological as sync
from syncological import ei2
from fakespikes import neurons, util, rates
from docopt import docopt
from joblib import Parallel, delayed


if __name__ == "__main__":
    args = docopt(__doc__, version='1.0')

    save_path = args['PATH']
    
    # Start at 0, or restart?
    k0 = 0
    if args['--restart_k']:
        k0 = int(args['--restart_k'])
    
    k = int(args['K'])
    
    n_stim = int(args['--n_stim'])

    conn_seed = None
    if args['--conn_seed']:
        conn_seed = int(args['--conn_seed'])
    
    stim_seed = int(args['--stim_seed'])
    prng = np.random.RandomState(stim_seed)

    balanced = True
    if args["--no_balanced"]:
        balanced = False

    # ------------------------------------------------------------
    # Params
    # -- Fixed
    # NOTE: expand CLI API to mod these, if needed in the future
    I_e = 0.0
    I_i = 0.0
    if args['--ing']:
        I_i = 0.8

    # -- Random
    codes = range(k0, k)
    w_es = prng.uniform(2, 16.0, k)[k0:k]  # Less than 3 means no spiking
    w_is = prng.uniform(2, 16.0, k)[k0:k]
    w_ees = prng.uniform(1.0, 8.0, k)[k0:k]
    w_iis = prng.uniform(1.0, 8.0, k)[k0:k]
    w_eis = prng.uniform(1.0, 16.0, k)[k0:k]
    w_ies = prng.uniform(1.0, 16.0, k)[k0:k]
    params = zip(codes, w_es, w_is, w_ees, w_iis, w_eis, w_ies)
    
    np.savez(os.path.join(save_path, "params"),
            codes=codes, w_es=w_es, w_is=w_is, w_ies=w_ies, I_e=I_e,
            w_ees=w_ees, w_iis=w_iis, w_eis=w_eis, I_i=I_i)

    # ------------------------------------------------------------
    # Create input
    # Load v1 rate data (1 ms resoultion)
    dt = 1e-3
    v1 = np.load(os.path.join(sync.__path__[0], 'data', 'no_opto_rates.npz'))
    stim = v1['rate']
    stim_times = v1['times']

    # Select 1-3 seconds, the visual stimulation period
    m = np.logical_and(stim_times > 1, stim_times <= 3)
    stim = stim[m]
    stim_times = stim_times[m]

    # Renorm stim_times to 0 + 0.1 s
    stim_times -= stim_times.min()
    stim_times += 0.1

    # Pad with 0.1 s of zeros
    stim_times = np.concatenate([
            np.zeros(int(np.ceil(0.1 * (1 / dt))) - 1), 
            stim_times
        ])
    stim = np.concatenate([
            np.zeros(int(np.ceil(0.1 * (1 / dt))) - 1), 
            stim
        ])

    # Create Poisson firing, mocking up
    # the stimulus.
    time = stim_times.max()
    nrns = neurons.Spikes(n_stim, time, dt=dt, seed=stim_seed)

    z = 5  # rate multplier (very few neurons in this model)
    spks_stim = nrns.poisson(z * stim)

    ns, ts = util.to_spiketimes(nrns.times, spks_stim) 

    # save input
    np.savez(os.path.join(save_path, "input"), 
            time=time, stim=stim, ns=ns, ts=ts, 
            stim_seed=stim_seed)

    # ------------------------------------------------------------
    # Run
    Parallel(n_jobs=int(args['--n_job']), verbose=3)(
        delayed(ei2.model)(
            os.path.join(save_path, str(code)), 
            time, n_stim, ts, ns, 
            w_e, w_i, w_ei, w_ie, w_ee, w_ii,
            I_e=I_e, I_i=I_i,
            verbose=False, parallel=True, 
            seed=stim_seed, conn_seed=conn_seed) 
        for code, w_e, w_i, w_ee, w_ii, w_ei, w_ie in params
    )
