"""Usage: ei2.py PATH K 
    (--ing | --ping)
    [--stim_seed=STIM_SEED]
    [--n_stim=NSTIM]
    [--n_jobs=NJOBS]

Simulate a EI circuit with HH neurons.

    Arguments:
        PATH        path to the results files
        K           number of simulations to run

    Options:
        -h --help                   show this screen
        --ing                       run in ING mode
        --ping                      run in PING mode
        --stim_seed=STIM_SEED       seed for creating the stimulus [default: 42]
        --n_stim=NSTIM              number of driving neurons [default: 100]
        --n_jobs=NJOBS              number of parallel jobs [default: 10]

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
    k = int(args['K'])
    stim_seed = int(args['--stim_seed'])
    n_stim = int(args['--n_stim'])
 
    prng = np.random.RandomState(stim_seed)

    # ------------------------------------------------------------
    # Params
    # -- Fixed
    # NOTE: expand CLI API to mod these, if needed in the future
    w_i = 0.5
    w_ee = 0.5
    w_ii = 0.5
    w_ei = 1.0

    I_e = 0.1
    I_i = 0.1
    if args['--ing']:
        I_i = 0.8

    # -- Random
    codes = range(k)
    w_es = prng.uniform(2, 10, k)
    w_ies = prng.uniform(0.1, 3.0, k)
    params = zip(codes, w_es, w_ies)
    
    np.savez(os.path.join(save_path, "params"),
            codes=codes, w_es=w_es, w_ies=w_ies, I_e=I_e,
            w_i=w_i, w_ee=w_ee, w_ii=w_ii, w_ei=w_ei, I_i=I_i)

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
    Parallel(n_jobs=int(args['--n_jobs']), verbose=3)(
        delayed(ei2.model)(
            os.path.join(save_path, str(code)), 
            time, n_stim, ts, ns, 
            w_e, w_i, w_ei, w_ie, w_ee, w_ii,
            I_e=I_e, I_i=I_i,
            verbose=False, parallel=True, 
            seed=stim_seed) 
        for code, w_e, w_ie in params
    )
