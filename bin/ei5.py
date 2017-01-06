"""Usage: ei5.py PATH K 
    (--ing | --ping)
    [--analysis_only]
    [--run_only]
    [--I_e=I_E]
    [--w_e=W_E]
    [--w_i=W_I]
    [--w_ee=W_EE]
    [--w_ii=W_II]
    [--w_ei=W_EI]
    [--w_ie=W_IE]
    [--scale_noise=NOISE]
    [--scale_rate=RATE]
    [--stim_seed=STIM_SEED]
    [--conn_seed=CONN_SEED]
    [--n_stim=NSTIM]
    [--n_job=NJOB]
    [--restart_k=RK]
    [--verbose]

Simulate a EI circuit with HH neurons, searching the network connectivity space.

    Arguments:
        PATH        path to the results files
        K           number of simulations to run
        
    Options:
        -h --help                   show this screen
        --analysis_only             (re)run the post-stimulation analysis
        --run_only                  skip the post-run analysis
        --ing                       run in ING mode
        --ping                      run in PING mode
        --I_e=I_E                   constant bias ('Vm tone') [default: 0.0]
        --w_e=W_E                   stim to E connection weight [default: 2.0]
        --w_i=W_I                   stim to I connection weight [default: 0.0]
        --w_ee=W_EE                 E to E connection weight [default: 1.0]
        --w_ii=W_II                 I to I connection weight [default: 1.0]
        --w_ei=W_EI                 E to I connection weight [default: 2.0]
        --w_ie=W_IE                 I to E connection weight [default: 8.0]
        --scale_noise=NOISE         scale variance in background [default: 4]  
        --scale_rate=RATE           scale input firing rate [default: 4]  
        --n_stim=NSTIM              stimulus population size [default: 100]
        --stim_seed=STIM_SEED       seed for creating the stimulus [default: 42]
        --conn_seed=CONN_SEED       initial seed for connections [default: 13]
        --n_job=NJOB                number of parallel jobs [default: 1]
        --verbose                   use verbose run mode
"""
from __future__ import division

import numpy as np
import os, sys, csv
from docopt import docopt
from joblib import Parallel, delayed

import syncological as sync
from syncological import ei5
from syncological.inputs import pad_rate, create_trials
from syncological.results import load_exp, load_trial, stim_seperation
from syncological.results import (trial_coding, trial_kl, power_coding,
                                  phase_coding, trial_S)
from syncological.results import extract_lfps, gamma_amplitude, peak_troughs
from fakespikes import neurons, util, rates


def analyze_results(data_path, n_trials, n_neurons=10, offset=0.1):
    # -- Time info
    fs = int(np.round(1 / 1e-5))
    dt = 1 / float(fs)
    times = load_trial(1, 0, data_path,
                       to_load=['exampletrace_e'])['exampletrace_e'][:, 0]
    times = times[times > offset]
    t_range = (times.min(), times.max())

    # - Get data
    to_load = ["spiketimes_e", "spiketimes_stim", "lfp", "exampletrace_e"]

    vis_1 = load_exp(1, n_trials, data_path, to_load, offset=offset)
    vis_2 = load_exp(2, n_trials, data_path, to_load, offset=offset)
    # box_1 = load_exp(3, n_trials, data_path, to_load, offset=offset)
    # box_2 = load_exp(4, n_trials, data_path, to_load, offset=offset)

    # ==================================================================
    # STIMULUS SEPERATION
    # Tell two stimuli apart, difference is in spike timing not 
    # ==================================================================

    analysis_name = "stim"
    save_path = os.path.join(data_path, analysis_name)
    try:
        os.makedirs(save_path)
    except OSError:
        if not os.path.isdir(save_path):
            raise

    vis_seps, vis_fracs = stim_seperation(vis_1, vis_2, dt)
    # box_seps, box_fracs = stim_seperation(box_1, box_2, dt)

    # save
    np.savetxt(
        os.path.join(save_path, "vis_sep.csv"),
        np.vstack(vis_seps).transpose(),
        delimiter=",",
        fmt='%.6f')
    np.savetxt(
        os.path.join(save_path, "vis_frac.csv"),
        vis_fracs,
        delimiter=",",
        fmt='%.6f')
    # np.savetxt(
    #     os.path.join(save_path, "box_sep.csv"),
    #     np.vstack(box_seps).transpose(),
    #     delimiter=",",
    #     fmt='%.6f')
    # np.savetxt(
    #     os.path.join(save_path, "box_frac.csv"),
    #     box_fracs,
    #     delimiter=",",
    #     fmt='%.6f')

    # ==================================================================
    # INFORMAION TRANSMISSION
    # Using the stimulus spike train as the reference.
    # ==================================================================

    # - Coding over all trials
    vis_precs, vis_lev_s, vis_lev_o, vis_lev_r = trial_coding(vis_1,
                                                              ref='stim')
    # box_precs, box_lev_s, box_lev_o, box_lev_r = trial_coding(box_1,
    #                                                           ref='stim')

    np.savetxt(
        os.path.join(save_path, "vis_trial_coding.csv"),
        np.vstack([vis_precs, vis_lev_s, vis_lev_o, vis_lev_r]).transpose(),
        header="prec,spike,order,rate",
        delimiter=",",
        fmt='%.6f')
    # np.savetxt(
    #     os.path.join(save_path, "box_trial_coding.csv"),
    #     np.vstack([box_precs, box_lev_s, box_lev_o, box_lev_r]).transpose(),
    #     header="prec,spike,order,rate",
    #     delimiter=",",
    #     fmt='%.6f')

    # - KL divergence over all trials
    vis_kl_s, vis_kl_o, vis_kl_r = trial_kl(vis_1, ref='stim')
    # box_kl_s, box_kl_o, box_kl_r = trial_kl(box_1, ref='stim')

    np.savetxt(
        os.path.join(save_path, "vis_trial_kl.csv"),
        np.vstack([vis_kl_s, vis_kl_o, vis_kl_r]).transpose(),
        header="spike,order,rate",
        delimiter=",",
        fmt='%.6f')
    # np.savetxt(
    #     os.path.join(save_path, "box_trial_kl.csv"),
    #     np.vstack([box_kl_s, box_kl_o, box_kl_r]).transpose(),
    #     header="spike,order,rate",
    #     delimiter=",",
    #     fmt='%.6f')

    # - Synch
    vis_isi, vis_sync = trial_S(vis_1, ref='stim')
    # box_isi, box_sync = trial_S(box_1, ref='stim')
    np.savetxt(
        os.path.join(save_path, "vis_trial_S.csv"),
        np.vstack([vis_isi, vis_sync]).transpose(),
        header="isi,sync",
        delimiter=",",
        fmt='%.6f')
    # np.savetxt(
    #     os.path.join(save_path, "box_trial_S.csv"),
    #     np.vstack([box_isi, box_sync]).transpose(),
    #     header="isi,sync",
    #     delimiter=",",
    #     fmt='%.6f')

    # - LFP depcomposition
    vis_lfps = extract_lfps(vis_1)
    vis_amps = gamma_amplitude(vis_lfps, fs)
    vis_peaks, vis_troughs = peak_troughs(vis_lfps, fs)

    # box_lfps = extract_lfps(box_1)
    # box_amps = gamma_amplitude(box_lfps, fs)
    # box_peaks, box_troughs = peak_troughs(box_lfps, fs)

    # - Coding as gamma power fluctutates.
    (vis_precs, vis_lev_s, vis_lev_o, vis_lev_r, vis_pows, vis_trials, skip_i,
     skip_r) = power_coding(vis_1,
                            times,
                            vis_lfps,
                            vis_amps,
                            vis_troughs,
                            ref='stim',
                            relative=True)

    # (box_precs, box_lev_s, box_lev_o, box_lev_r, box_pows, box_trials, skip_i,
    #  skip_r) = power_coding(box_1,
    #                         times,
    #                         box_lfps,
    #                         box_amps,
    #                         box_troughs,
    #                         ref='stim',
    #                         relative=True)

    np.savetxt(
        os.path.join(save_path, "vis_pow.csv"),
        np.vstack([vis_trials, vis_pows, vis_precs, vis_lev_s, vis_lev_o,
                   vis_lev_r]).transpose(),
        header="trials,amp,prec,spike,order,rate",
        delimiter=",",
        fmt='%.6f')
    # np.savetxt(
    #     os.path.join(save_path, "box_pow.csv"),
    #     np.vstack([box_trials, box_pows, box_precs, box_lev_s, box_lev_o,
    #                box_lev_r]).transpose(),
    #     header="trials,amp,prec,spike,order,rate",
    #     delimiter=",",
    #     fmt='%.6f')

    # - Coding with phase
    (vis_precs_rise, vis_precs_fall, vis_lev_s_rise, vis_lev_s_fall,
     vis_lev_o_rise, vis_lev_o_fall, vis_lev_r_rise, vis_lev_r_fall,
     vis_trials_k, skip_f, skip_r) = phase_coding(vis_1,
                                                  times,
                                                  vis_peaks,
                                                  vis_troughs,
                                                  ref='stim',
                                                  relative=True)

    # (box_precs_rise, box_precs_fall, box_lev_s_rise, box_lev_s_fall,
    #  box_lev_o_rise, box_lev_o_fall, box_lev_r_rise, box_lev_r_fall,
    #  box_trials_k, skip_f, skip_r) = phase_coding(box_1,
    #                                               times,
    #                                               box_peaks,
    #                                               box_troughs,
    #                                               ref='stim',
    #                                               relative=True)
    
    np.savetxt(
        os.path.join(save_path, "vis_rise.csv"),
        np.vstack([vis_trials_k, vis_precs_rise, vis_lev_s_rise,
                   vis_lev_o_rise, vis_lev_r_rise]).transpose(),
        header="trials,prec,spike,order,rate",
        delimiter=",",
        fmt='%.6f')
    np.savetxt(
        os.path.join(save_path, "vis_fall.csv"),
        np.vstack([vis_trials_k, vis_precs_fall, vis_lev_s_fall,
                   vis_lev_o_fall, vis_lev_r_fall]).transpose(),
        header="trials,prec,spike,order,rate",
        delimiter=",",
        fmt='%.6f')
    # np.savetxt(
    #     os.path.join(save_path, "box_rise.csv"),
    #     np.vstack([box_trials_k, box_precs_rise, box_lev_s_rise,
    #                box_lev_o_rise, box_lev_r_rise]).transpose(),
    #     header="trials,prec,spike,order,rate",
    #     delimiter=",",
    #     fmt='%.6f')
    # np.savetxt(
    #     os.path.join(save_path, "box_fall.csv"),
    #     np.vstack([box_trials_k, box_precs_fall, box_lev_s_fall,
    #                box_lev_o_fall, box_lev_r_fall]).transpose(),
    #     header="trials,prec,spike,order,rate",
    #     delimiter=",",
    #     fmt='%.6f')

    # ==================================================================
    # TRIAL CONSISTENCY
    # How does coding vary by trial, with respect to the first trial 
    # ==================================================================

    analysis_name = "trials"
    save_path = os.path.join(data_path, analysis_name)
    try:
        os.makedirs(save_path)
    except OSError as err:
        if not os.path.isdir(save_path):
            raise

    # - Coding over all trials
    vis_precs, vis_lev_s, vis_lev_o, vis_lev_r = trial_coding(vis_1, ref='e')
    # box_precs, box_lev_s, box_lev_o, box_lev_r = trial_coding(box_1, ref='e')

    np.savetxt(
        os.path.join(save_path, "vis_trial_coding.csv"),
        np.vstack([vis_precs, vis_lev_s, vis_lev_o, vis_lev_r]).transpose(),
        header="prec,spike,order,rate",
        delimiter=",",
        fmt='%.6f')
    # np.savetxt(
    #     os.path.join(save_path, "box_trial_coding.csv"),
    #     np.vstack([box_precs, box_lev_s, box_lev_o, box_lev_r]).transpose(),
    #     header="prec,spike,order,rate",
    #     delimiter=",",
    #     fmt='%.6f')

    # - KL divergence over all trials
    vis_kl_s, vis_kl_o, vis_kl_r = trial_kl(vis_1, ref='e')
    # box_kl_s, box_kl_o, box_kl_r = trial_kl(box_1, ref='e')

    np.savetxt(
        os.path.join(save_path, "vis_trial_kl.csv"),
        np.vstack([vis_kl_s, vis_kl_o, vis_kl_r]).transpose(),
        header="spike,order,rate",
        delimiter=",",
        fmt='%.6f')
    # np.savetxt(
    #     os.path.join(save_path, "box_trial_kl.csv"),
    #     np.vstack([box_kl_s, box_kl_o, box_kl_r]).transpose(),
    #     header="spike,order,rate",
    #     delimiter=",",
    #     fmt='%.6f')

    # - Coding as gamma power fluctutates.
    # (vis_precs, vis_lev_s, vis_lev_o, vis_lev_r, vis_pows, vis_trials, skip_i,
    #  skip_r) = power_coding(vis_1,
    #                         times,
    #                         vis_lfps,
    #                         vis_amps,
    #                         vis_troughs,
    #                         ref='stim',
    #                         relative=True)

    # (box_precs, box_lev_s, box_lev_o, box_lev_r, box_pows, box_trials, skip_i,
    #  skip_r) = power_coding(box_1,
    #                         times,
    #                         box_lfps,
    #                         box_amps,
    #                         box_troughs,
    #                         ref='stim',
    #                         relative=True)

    # np.savetxt(
    #     os.path.join(save_path, "vis_pow.csv"),
    #     np.vstack([vis_trials, vis_pows, vis_precs, vis_lev_s, vis_lev_o,
    #                vis_lev_r]).transpose(),
    #     header="trials,amp,prec,spike,order,rate",
    #     delimiter=",",
    #     fmt='%.6f')
    # np.savetxt(
    #     os.path.join(save_path, "box_pow.csv"),
    #     np.vstack([box_trials, box_pows, box_precs, box_lev_s, box_lev_o,
    #                box_lev_r]).transpose(),
    #     header="trials,amp,prec,spike,order,rate",
    #     delimiter=",",
    #     fmt='%.6f')

    # # - Coding with phase
    # (vis_precs_rise, vis_precs_fall, vis_lev_s_rise, vis_lev_s_fall,
    #  vis_lev_o_rise, vis_lev_o_fall, vis_lev_r_rise, vis_lev_r_fall,
    #  vis_trials_k, skip_f, skip_r) = phase_coding(vis_1,
    #                                               times,
    #                                               vis_peaks,
    #                                               vis_troughs,
    #                                               ref='stim',
    #                                               relative=True)

    # (box_precs_rise, box_precs_fall, box_lev_s_rise, box_lev_s_fall,
    #  box_lev_o_rise, box_lev_o_fall, box_lev_r_rise, box_lev_r_fall,
    #  box_trials_k, skip_f, skip_r) = phase_coding(box_1,
    #                                               times,
    #                                               box_peaks,
    #                                               box_troughs,
    #                                               ref='stim',
    #                                               relative=True)

    # np.savetxt(
    #     os.path.join(save_path, "vis_rise.csv"),
    #     np.vstack([vis_trials_k, vis_precs_rise, vis_lev_s_rise,
    #                vis_lev_o_rise, vis_lev_r_rise]).transpose(),
    #     header="trials,prec,spike,order,rate",
    #     delimiter=",",
    #     fmt='%.6f')
    # np.savetxt(
    #     os.path.join(save_path, "vis_fall.csv"),
    #     np.vstack([vis_trials_k, vis_precs_fall, vis_lev_s_fall,
    #                vis_lev_o_fall, vis_lev_r_fall]).transpose(),
    #     header="trials,prec,spike,order,rate",
    #     delimiter=",",
    #     fmt='%.6f')
    # np.savetxt(
    #     os.path.join(save_path, "box_rise.csv"),
    #     np.vstack([box_trials_k, box_precs_rise, box_lev_s_rise,
    #                box_lev_o_rise, box_lev_r_rise]).transpose(),
    #     header="trials,prec,spike,order,rate",
    #     delimiter=",",
    #     fmt='%.6f')
    # np.savetxt(
    #     os.path.join(save_path, "box_fall.csv"),
    #     np.vstack([box_trials_k, box_precs_fall, box_lev_s_fall,
    #                box_lev_o_fall, box_lev_r_fall]).transpose(),
    #     header="trials,prec,spike,order,rate",
    #     delimiter=",",
    #     fmt='%.6f')


if __name__ == "__main__":
    args = docopt(__doc__, version='1.0')

    verbose = False
    if args['--verbose']:
        verbose = True

    # ------------------------------------------------------------
    # Handle flags
    conn_seed = int(args['--conn_seed'])
    stim_seed = int(args['--stim_seed'])
    prng = np.random.RandomState(stim_seed)

    save_path = args['PATH']
    k = int(args['K'])
    scale_noise = float(args['--scale_noise'])
    scale_rate = float(args['--scale_rate'])
    n_stim = int(args['--n_stim'])

    I_e = float(args['--I_e'])
    I_i = 0.0
    if args['--ing']:
        I_i = 0.8

    w_e = float(args['--w_e'])
    w_i = float(args['--w_i'])

    w_ie = float(args['--w_ie'])
    w_ei = float(args['--w_ei'])
    w_ee = float(args['--w_ee'])
    w_ii = float(args['--w_ii'])

    # ------------------------------------------------------------
    # Init input (params are generated below)
    dt = 1e-3

    # - Define v1-like rates
    v1_data = np.load(os.path.join(sync.__path__[0], 'data',
                                   'no_opto_rates.npz'))
    v1 = v1_data['rate']
    v1_times = v1_data['times'][:-1]
    assert v1.shape == v1_times.shape, "v1 shape problem {}, {}".format(
        v1.shape, v1_times.shape)

    # Select 1-3 seconds, the visual stimulation period
    m = np.logical_and(v1_times > 1, v1_times <= 3)
    v1 = v1[m]
    v1_times = v1_times[m]

    # Renorm v1_times to 0 + 0.1 s
    v1_times -= v1_times.min()
    v1, v1_times = pad_rate(0.1, v1, v1_times, dt)
    time = v1_times.max()

    # instance 1
    nrns = neurons.Spikes(n_stim, time, dt=dt, seed=stim_seed)
    spks_v1 = nrns.poisson(scale_rate * v1)
    ns_v1_1, ts_v1_1 = util.to_spiketimes(nrns.times, spks_v1)
    ns_v1_1, ts_v1_1 = create_trials(k, 0, ns_v1_1, ts_v1_1)

    # instance 2
    nrns = neurons.Spikes(n_stim, time, dt=dt, seed=stim_seed + 99394)
    spks_v1 = nrns.poisson(scale_rate * v1)
    ns_v1_2, ts_v1_2 = util.to_spiketimes(nrns.times, spks_v1)
    ns_v1_2, ts_v1_2 = create_trials(k, 0, ns_v1_2, ts_v1_2)

    # - Define boxcar
    box_times = neurons.Spikes(2, time, dt=dt).times
    box = rates.constant(box_times, np.mean(v1))
    box, box_times = pad_rate(0.1, box, box_times, dt)

    # instance 1
    nrns = neurons.Spikes(n_stim, box_times.max() - dt, dt=dt, seed=stim_seed)
    spks_box = nrns.poisson(scale_rate * box)
    ns_box_1, ts_box_1 = util.to_spiketimes(box_times, spks_box)
    ns_box_1, ts_box_1 = create_trials(k, 0, ns_box_1, ts_box_1)

    # instance 2
    nrns = neurons.Spikes(n_stim,
                          box_times.max() - dt,
                          dt=dt,
                          seed=stim_seed + 99394)
    spks_box = nrns.poisson(scale_rate * box)
    ns_box_2, ts_box_2 = util.to_spiketimes(box_times, spks_box)
    ns_box_2, ts_box_2 = create_trials(k, 0, ns_box_2, ts_box_2)

    inputs = [
        (1, ns_v1_1, ts_v1_1), (2, ns_v1_2, ts_v1_2), (3, ns_box_1, ts_box_1),
        (4, ns_box_2, ts_box_2)
    ]

    # ------------------------------------------------------------
    # Run
    if not args['--analysis_only']:
        # Run
        Parallel(n_jobs=int(args['--n_job']),
                 verbose=3)(delayed(ei5.model)(
                     os.path.join(save_path, str(code)),
                     time,
                     k,
                     n_stim,
                     ts,
                     ns,
                     w_e,
                     w_i,
                     w_ei,
                     w_ie,
                     w_ee,
                     w_ii,
                     I_e=I_e,
                     I_i=I_i,
                     verbose=verbose,
                     parallel=True,
                     scale_noise=scale_noise,
                     seed=stim_seed,
                     conn_seed=conn_seed) for code, ns, ts in inputs)

    # Anaysis
    if not args['--run_only']:
        analyze_results(save_path, k)
