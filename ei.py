#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Implements a sparse PING E-I model, based Borges et al PNAS 2005.
"""
from __future__ import division
import argparse
import numpy as np
from brian2 import *
from syncological.inputs import gaussian_impulse
from scipy.stats.mstats import zscore
from foof.util import create_psd
from fakespikes import util as futil
import pyspike as spk


def model(name, time, N_stim, ts_stim, idx_stim, period,
          w_e, w_i, w_ei, w_ie, w_ee, w_ii,
          I_e, I_i, I_i_sigma, I_e_sigma,
          stdp, seed=None):
    """Model some BRAINS!"""

    np.random.seed(seed)

    # Reconile time and period
    if np.allclose(time, period):
        N_trials = 1
    elif np.allclose(time % period, 0):
        N_trials = int(time / period) 
    else:
        raise ValueError("time must be an integer multiple of period")

    time = time * second
    period = period * second

    time_step = 0.01 * ms
    defaultclock.dt = time_step

    # time += time_step  # SpikeGeneratorGroup compatibility
    # period += time_step

    # --
    # Model
    # Network
    N = 400
    N_e = int(N * 0.8)
    N_i = int(N * 0.2)

    r_e = 0 * Hz
    r_i = r_e

    delay = 2 * ms
    p_ei = 0.1
    p_ie = 0.1
    p_e = 0.1
    p_ii = 0.6

    w_e = w_e / (p_e * N_e) * msiemens
    w_i = w_i * msiemens
    w_ei = w_ei / (p_ei * N_e) * msiemens
    w_ie = w_ie / (p_ie * N_i) * msiemens
    w_ee = w_ee / (p_e * N_e) * msiemens
    w_ii = w_ii / (p_ii * N_i) * msiemens
    w_m = 0 / N_e * msiemens  # Read ref 47 to get value

    # --
    # Fixed
    # cell
    Cm = 1 * uF  # /cm2

    g_Na = 100 * msiemens
    g_K = 80 * msiemens
    g_l = 0.1 * msiemens

    V_Na = 50 * mV
    V_K = -100 * mV
    V_l = -67 * mV

    # syn
    tau_r_ampa = 0.2 * ms
    tau_d_ampa = 2 * ms
    tau_r_gaba = 0.5 * ms
    tau_d_gaba = 10 * ms
    tau_r_nmda = 1 * ms
    tau_d_nmda = 100 * ms

    V_thresh = 20 * mV
    V_i = -80 * mV
    V_e = 0 * mV

    hh = """
    dV/dt = (I_Na + I_K + I_l + I_m + I_syn + I) / Cm : volt
    """ + """
    I_Na = g_Na * (m ** 3) * h * (V_Na - V) : amp
    m = a_m / (a_m + b_m) : 1
    a_m = (0.32 * (54 + V/mV)) / (1 - exp(-0.25 * (V/mV + 54))) / ms : Hz
    b_m = (0.28 * (27 + V/mV)) / (exp(0.2 * (V/mV + 27)) - 1) / ms : Hz
    h = clip(1 - 1.25 * n, 0, inf) : 1
    """ + """
    I_K = g_K * n ** 4 * (V_K - V) : amp
    dn/dt = (a_n - (a_n * n)) - b_n * n : 1
    a_n = (0.032 * (52 + V/mV)) / (1 - exp(-0.2 * (V/mV + 52))) / ms : Hz
    b_n = 0.5 * exp(-0.025 * (57 + V/mV)) / ms : Hz
    """ + """
    I_l = g_l * (V_l - V) : amp
    """ + """
    I_m = w_m * w * (V_K - V) : amp
    dw/dt = (w_inf - w) / tau_w/ms : 1
    w_inf = 1 / (1 + exp(-1 * (V/mV + 35) / 10)) : 1
    tau_w = 400 / ((3.3 * exp((V/mV + 35)/20)) + (exp(-1 * (V/mV + 35) / 20))) : 1
    """ + """
    I_syn = g_e * (V_e - V) +
        g_ee * (V_e - V) +
        g_i * (V_i - V)  + 
        g_s * (V_e - V) : amp
    dg_e/dt = -g_e / tau_d_ampa : siemens
    dg_ee/dt = -g_ee / tau_d_ampa : siemens
    dg_i/dt = -g_i / tau_d_gaba : siemens
    dg_s/dt = -g_s / tau_d_ampa : siemens
    I : amp
    """

    P_e = NeuronGroup(
        N_e, model=hh,
        threshold='V >= V_thresh',
        refractory=3 * ms
    )

    P_i = NeuronGroup(
        N_i,
        model=hh,
        threshold='V >= V_thresh',
        refractory=3 * ms
    )

    P_e.V = 'randn() * 0.1 * V_l'
    P_i.V = 'randn() * 0.1 * V_l'

    P_e.g_e = 'randn() * 0.1 * w_e'
    P_e.g_ee = 'randn() * 0.1 * w_ee'
    P_e.g_i = 'randn() * 0.1 * w_ie'
    P_i.g_i = 'randn() * 0.1 * w_ii'

    if np.allclose(I_e_sigma, 0.0):
        P_e.I = I_e * uamp
    else:
        P_e.I = np.random.normal(I_e, I_e_sigma, N_e) * uamp

    if np.allclose(I_i_sigma, 0.0):
        P_i.I = I_i * uamp
    else:
        P_i.I = np.random.normal(I_i, I_i_sigma, N_i) * uamp

    P_e_back = PoissonGroup(N_e, rates=r_e)
    P_i_back = PoissonGroup(N_i, rates=r_i)
    P_stim = SpikeGeneratorGroup(N_stim, idx_stim, ts_stim * second, 
            period=period)

    # --
    # Syn
    # Internal
    C_ei = Synapses(P_e, P_i, pre='g_e += w_ei', delay=delay)
    C_ei.connect(True, p=p_ei)

    C_ie = Synapses(P_i, P_e, pre='g_i += w_ie', delay=delay)
    C_ie.connect(True, p=p_ie)

    C_ii = Synapses(P_i, P_i, pre='g_i += w_ii', delay=delay)
    C_ii.connect(True, p=p_ii)

    C_ee = Synapses(P_e, P_e, pre='g_ee += w_ee', delay=delay)
    C_ee.connect(True, p=p_e)

    # External
    C_stim_e = Synapses(P_stim, P_e, pre='g_s += w_e')
    C_stim_e.connect(True, p=p_e)

    if stdp:
        stdp_syn =  """
        w_stdp : siemens
        dApre/dt = -Apre / tau_pre : siemens (event-driven)
        dApost/dt = -Apost / tau_post : siemens (event-driven)
        """
        tau_pre = 20 * ms
        tau_post = tau_pre

        # SE 
        gmax_e = w_e * 10
        dpre_e = 0.005
        dpost_e = -dpre_e * tau_pre / tau_post * 1.05
        dpre_e *= w_e
        dpost_e *= w_e
        
        C_stim_e = Synapses(
            P_stim, P_e, stdp_syn, 
            pre="""
            g_e += w_stdp
            Apre += dpre_e
            w_stdp = clip(w_stdp + dpost_e, 0, gmax_e)
            """,
            post="""
            Apost += dpost_e
            w_stdp = clip(w_stdp + dpre_e, 0, gmax_e)
            """
        )
        C_stim_e.connect(True, p=p_e) 
        C_stim_e.w_stdp = 'w_e + (randn() * 0.1 * w_e)'
        
        # EE 
        gmax_ee = w_ee * 10
        dpre_ee = 0.005
        dpost_ee = -dpre_ee * tau_pre / tau_post * 1.05
        dpre_ee *= w_ee
        dpost_ee *= w_ee
        
        C_ee = Synapses(
            P_e, P_e, stdp_syn, 
            pre="""
            g_ee += w_stdp
            Apre += dpre_ee
            w_stdp = clip(w_stdp + dpost_ee, 0, gmax_ee)
            """,
            post="""
            Apost += dpost_ee
            w_stdp = clip(w_stdp + dpre_ee, 0, gmax_ee)
            """
        )
        C_ee.connect(True, p=p_e)
        C_ee.w_stdp = 'w_ee + (randn() * 0.1 * w_ee)'

    # --
    # Create network and save
    net = Network(
        P_e, P_i, P_e_back, P_i_back, P_stim, 
        C_ee, C_ii, C_ie, C_ei, C_stim_e
    )
    net.store('trials')

    # --
    # Go!
    # Loop over trials (if present)
    results = []
    t_last = 0
    for k in range(N_trials):
        print("Trial {0}".format(k))

        net.restore('trials')

        # Add fresh monitors before run
        spikes_i = SpikeMonitor(P_i)
        spikes_e = SpikeMonitor(P_e)
        spikes_stim = SpikeMonitor(P_stim)
        pop_stim = PopulationRateMonitor(P_stim)
        pop_e = PopulationRateMonitor(P_e)
        pop_i = PopulationRateMonitor(P_i)
        weights_ee = StateMonitor(C_ee, 'w_stdp', record=True)
        weights_e = StateMonitor(C_stim_e, 'w_stdp', record=True)
        traces_e = StateMonitor(P_e, ('V', 'g_e', 'g_i', 'g_s', 'g_ee'),
                                record=True)
        traces_i = StateMonitor(P_i, ('V', 'g_e', 'g_i'),
                                record=range(11, 31))

        monitors = [spikes_i, spikes_e, spikes_stim,
                    pop_stim, pop_e, pop_i, traces_e, traces_i,
                    weights_e, weights_ee]
        net.add(monitors)
        net.run(period, report='text')

        # Then save and analyze the results, discarding the
        # (now stale) monitors
        print("Saving results")
        result = {
            'spikes_i': spikes_i,
            'spikes_e': spikes_e,
            'spikes_stim': spikes_stim,
            'pop_e': pop_e,
            'pop_stim': pop_stim,
            'pop_i': pop_i,
            'traces_e': traces_e,
            'traces_i': traces_i,
            'weights_e': weights_e,
            'weights_ee' : weights_ee
        }
 
        trial_name = name + "_trial-" + str(k)
        save_result(trial_name, result)
        analyze_result(trial_name, result, fs=100000, save=True)

        net.remove(monitors)
        net.store('trials')

    return result  # return the last result only


def save_result(name, result, fs=10000):
    spikes_e = result['spikes_e']
    spikes_i = result['spikes_i']
    spikes_stim = result['spikes_i']
    pop_e = result['pop_e']
    pop_i = result['pop_i']
    traces_e = result['traces_e']
    traces_i = result['traces_i']

    # --
    # Save full
    # Spikes
    np.savetxt(name + '_spiketimes_e.csv',
               np.vstack([spikes_e.i, spikes_e.t, ]).transpose(),
               fmt='%.i, %.5f')
    np.savetxt(name + '_spiketimes_i.csv',
               np.vstack([spikes_i.i, spikes_i.t, ]).transpose(),
               fmt='%.i, %.5f')
    np.savetxt(name + '_spikets_stim.csv',
               np.vstack([spikes_stim.i, spikes_stim.t, ]).transpose(),
               fmt='%.i, %.5f')

    # Example trace
    np.savetxt(name + '_exampletrace_e.csv',
               np.vstack([traces_e.t, traces_e.V[0], ]).transpose(),
               fmt='%.5f, %.4f')
    np.savetxt(name + '_exampletrace_i.csv',
               np.vstack([traces_i.t, traces_i.V[0], ]).transpose(),
               fmt='%.5f, %.4f')

    # Pop rate
    np.savetxt(name + '_poprates_e.csv',
               np.vstack([pop_e.t, pop_e.rate / Hz, ]).transpose(),
               fmt='%.5f, %.1f')
    np.savetxt(name + '_poprates_i.csv',
               np.vstack([pop_i.t, pop_i.rate / Hz, ]).transpose(),
               fmt='%.5f, %.1f')

    # TODO save weights
    # and neuron indices for weight can become a (i, j) matrix
    # LFP
    lfp = (np.abs(traces_e.g_e.sum(0)) +
           np.abs(traces_e.g_i.sum(0)) +
           np.abs(traces_e.g_ee.sum(0)))
    np.savetxt(name + '_lfp.csv', zscore(lfp), fmt='%.2f')

    # PSD
    lfp = lfp[:]  # Drop initial spike
    freqs, spec = create_psd(lfp, fs)
    np.savetxt(name + '_psd.csv', np.vstack(
        [freqs, np.log10(spec)]).transpose(), fmt='%.1f, %.3f')


# TODO Add/Update analysis window args
# TODO SFC?
def analyze_result(name, result, fs=100000, save=True):
    analysis = {}

    spikes_e = result['spikes_e']
    spikes_stim = result['spikes_stim']
    spikes_i = result['spikes_i']

    # Get Ns and ts
    ns_e, ts_e = spikes_e.i, spikes_e.t / second
    ns_i, ts_i = spikes_i.i, spikes_i.t / second
    ns_stim, ts_stim = spikes_stim.i, spikes_stim.t / second

    # # Keep only neurons 0-199
    # mask = ns_e < 200
    # ns_e, ts_e = ns_e[mask], ts_e[mask]
    # mask = ns_i < 200
    # ns_i, ts_i = ns_i[mask], ts_i[mask]
 
    # kappa
    r_e = futil.kappa(ns_e, ts_e, ns_e, ts_e, (0, 1), 1.0 / 1000)  # 1 ms bins
    analysis['kappa_e'] = r_e
    r_i = futil.kappa(ns_i, ts_i, ns_i, ts_i, (0, 1), 1.0 / 1000)  # 1 ms bins
    analysis['kappa_i'] = r_i

    # fano
    fanos_e = futil.fano(ns_e, ts_e)
    mfano_e = np.nanmean([x for x in fanos_e.values()])
    analysis['fano_e'] = mfano_e

    # ISI and SPIKE
    sto_e = spk.SpikeTrain(ts_e, (ts_e.min(), ts_e.max()))
    sto_stim = spk.SpikeTrain(ts_stim, (ts_stim.min(), ts_stim.max()))
    sto_e.sort()
    sto_stim.sort()
    isi = spk.isi_distance(sto_stim, sto_e)
    sync = spk.spike_sync(sto_stim, sto_e)
    analysis['isi_e'] = isi
    analysis['sync_e'] = sync

    # l distance and spike
    ordered_e, _ = futil.ts_sort(ns_e, ts_e)
    ordered_stim, _ = futil.ts_sort(ns_stim, ts_stim)
    lev = futil.levenshtein(list(ordered_stim), list(ordered_e))
    analysis['lev_e'] = lev

    if save:
        with open(name + '_analysis.csv', 'w') as f:
            [f.write('{0},{1}\n'.format(k, v)) for k, v in analysis.items()]

    # TODO SFC
    return analysis
