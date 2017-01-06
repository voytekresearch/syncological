"""Usage: net2.py PATH K 
    (--ing | --ping)
    [--w_e=W_E]
    [--w_i=W_I]
    [--w_ee=W_EE]
    [--w_ii=W_II]
    [--w_ei=W_EI]
    [--w_ie=W_IE]
    [--no_balanced]
    [--stim_seed=STIM_SEED]
    [--conn_seed=CONN_SEED]
    [--n_stim=NSTIM]
    [--n_job=NJOB]

Simulate a EI circuit with HH neurons, searching the network connectivity space.

    Arguments:
        PATH        path to the results files
        K           number of simulations to run

    Options:
        -h --help                   show this screen
        --ing                       run in ING mode
        --ping                      run in PING mode
        --w_e=W_E                   stim to E connection weight [default: 1.0]
        --w_i=W_I                   stim to I connection weight [default: 2.0]
        --w_ee=W_EE                 E to E connection weight [default: 1.0]
        --w_ii=W_II                 I to I connection weight [default: 1.0]
        --w_ei=W_EI                 E to I connection weight [default: 2.0]
        --w_ie=W_IE                 I to E connection weight [default: 3.0]
        --no_balanced               turn off balanced background activity     
        --stim_seed=STIM_SEED       seed for creating the stimulus [default: 42]
        --conn_seed=CONN_SEED       initial seed for connections [default: 13]
        --n_stim=NSTIM              number of driving neurons [default: 100]
        --n_job=NJOB                number of parallel jobs [default: 10]

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

    conn_seed = int(args['--conn_seed'])
    stim_seed = int(args['--stim_seed'])
    prng = np.random.RandomState(stim_seed)

    save_path = args['PATH']
    k = int(args['K'])
    n_stim = int(args['--n_stim'])
 
    balanced = True
    if args["--no_balanced"]:
        balanced = False

    # ------------------------------------------------------------
    # Params
    # -- Fixed
    # NOTE: expand CLI API to mod these, if needed in the future
    w_e = float(args['--w_e'])  # TODO tune me
    w_i = float(args['--w_i'])
    w_ie = float(args['--w_ie'])
    w_ei = float(args['--w_ei'])
    w_ee = float(args['--w_ee'])
    w_ii = float(args['--w_ii'])

    I_e = 0
    I_i = 0
    if args['--ing']:
        I_i = 0.8

    # -- Random
    codes = range(k)
    np.savez(os.path.join(save_path, "params"),
            codes=codes, w_es=w_e, w_ies=w_ie, I_e=I_e,
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
    Parallel(n_jobs=int(args['--n_job']), verbose=3)(
        delayed(ei2.model)(
            os.path.join(save_path, str(code)), 
            time, n_stim, ts, ns, 
            w_e, w_i, w_ei, w_ie, w_ee, w_ii,
            I_e=I_e, I_i=I_i,
            verbose=False, parallel=True, 
            seed=stim_seed, conn_seed=conn_seed + code) 
        for code in codes
    )
