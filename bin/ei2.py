"""Usage: ei2.py NAME K 
    (--ing | --ping)
    [--stim_seed=STIM_SEED]
    [--n_stim=NSTIM]


Simulate a EI circuit with HH neurons.

    Arguments:
        NAME        name (and path) of the results files
        K           number of simulations to run

    Options:
        -h --help                   show this screen
        --ing                       run in ING mode
        --ping                      run in PING mode
        --stim_seed=STIM_SEED       seed for creating the stimulus [default: 42]
        --n_stim=NSTIM              number of driving neurons [default: 100]

"""
from syncological import ei2
from fakespikes import neurons, util, rates


if __name__ == "__main__":
    args = docopt(__doc__, version='1.0')

    name = args['NAME']
    k = int(args['K'])
    stim_seed = int(args['--stim_seed'])
    n_stim = int(args['--n_stim'])
 
    # ------------------------------------------------------------
    # Params
    # -- Fixed
    time = 1
    w_i = 0.5
    w_ee = 1.0
    w_ii = 0.5
    w_ei = .5

    I_i = 0.1
    if args['--ing']:
        I_i = 0.8

    # -- Random
    # TODO init
    w_es = ( , )
    w_ies = ( , )
    I_es = ( , )

    # ------------------------------------------------------------
    # Create input
    dt = 0.001
    nrns = neurons.Spikes(n_stim, time, dt=dt, seed=seed)

    Istim = 20  # Avg rate of 'natural' stimulation
    Sstim = 0.001 * Istim  # Avg st dev of natural firing

    stim = rates.stim(nrns.times, Istim, Sstim, seed)

    spks_stim = nrns.poisson(stim)
    ns, ts = util.to_spiketimes(nrns.times, spks_stim)

    # Run
    # Joblib
