#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Implements a sparse ING E-I model, based Wang and Buzsaki[0].

[0]: Gamma oscillation by synaptic inhibition in a hippocampal interneuronal
network model. Wang XJ, Buzsaki G.J Neurosci. 1996 Oct 15;16(20):6402-13.
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


def model(time, time_stim, rate_stim, w_e, w_i, w_ie, I_e, I_i_sigma=0.01,
        seed=None):
    """Model some BRAINS!"""

    time = time * second
    time_stim = time_stim * second

    # Network
    N = 1000
    N_e = int(N * 0.8)
    N_i = int(N * 0.2)
    N_stim = int(N * 0.2)

    r_e = 10 * Hz
    r_i = r_e

    cdelay = 2 * ms

    w_e = w_e * msiemens
    w_i = w_i * msiemens
    w_ie = w_ie / N_i * msiemens

    w_ee = 0 / N_e * msiemens
    w_ii = 0.1 / N_i * msiemens
    w_m = 0 / N_e * msiemens  # Read ref 47 to get value

    # --
    # Model
    np.random.seed(seed)
    time_step = 0.1 * ms
    decimals = 4

    # --
    # cell biophysics
    # WB (I)
    I_i = 1.1  # * uamp
    I_i = np.random.normal(I_i, I_i_sigma, N_i) * uamp
    if I_e is None:
        I_e = (0.25, .25)

    # After-hyperpolaization (AHP)
    ahp = 5

    area_wb = 1 * cm ** 2
    Cm_wb = (1 * ufarad * cm ** -2) * area_wb
    Cm = 1 * uF  # /cm2

    gL_wb = (0.1 * msiemens * cm ** -2) * area_wb
    EL_wb = -70 * mV
    Vt_wb = -52 * mV
    ENa_wb = 55 * mV
    gNa_wb = (35 * msiemens * cm ** -2) * area_wb
    EK_wb = -90 * mV
    gK_wb = 9 * msiemens

    ESyn_wb = -75 * mV
    gSyn_wb = (0.1 / N_i) * msiemens

    # E
    taue_hh = 5 * ms
    taui_hh = 10 * ms
    win_hh = 26 * nsiemens

    # HH (E)
    Cm = 1 * uF  # /cm2

    g_Na = 100 * msiemens
    g_K = 80 * msiemens
    g_l = 0.1 * msiemens

    V_Na = 50 * mV
    V_K = -100 * mV
    V_l = -67 * mV
    V_thresh = 20 * mV

    # synapse biophysics
    tau_r_ampa = 0.2 * ms
    tau_d_ampa = 2 * ms
    tau_r_gaba = 0.5 * ms
    tau_d_gaba = 10 * ms
    tau_r_nmda = 1 * ms
    tau_d_nmda = 100 * ms

    V_i = -80 * mV
    V_e = 0 * mV

    # --
    # Eqs
    wb = """
    dv/dt = (-INa - IK - IL - I_syn + I) / Cm_wb : volt
    """ + """
    INa    = gNa_wb * m**3 * h * (v-ENa_wb) : amp
    m      = alpham / (alpham + betam) : 1
    dh/dt  = ahp * (alphah * (1 - h) - betah * h) : 1
    alpham = (-0.1/mV) * (v + 35*mV) / 
        (exp(-0.1/mV * (v + 35*mV)) - 1)/ms : Hz
    betam  = 4 * exp(-(v + 60*mV) / (18*mV))/ms : Hz
    alphah = 0.07 * exp(-(v + 58*mV) / (20*mV))/ms : Hz
    betah  = 1.0 / (exp(-0.1/mV * (v + 28*mV)) + 1)/ms : Hz
    """ + """
    IK     = gK_wb * n**4 * (v - EK_wb) : amp
    dn/dt  = ahp * (alphan * (1 - n) - (betan * n)) : 1
    alphan = -0.01/mV * (v + 34*mV) / 
        (exp(-0.1/mV * (v + 34*mV)) - 1)/ms : Hz
    betan  = 0.125 * exp(-(v + 44*mV) / (80*mV))/ms : Hz
    """ + """
    IL   = gL_wb * (v - EL_wb) : amp
    """ + """
    I_syn = g_e * (v - V_e) +
        g_i * (v - ESyn_wb) : amp
    g_e : siemens
    g_i : siemens
    """ + """
    I : amp
    theta : 1
    """

    hh = """
    dV/dt = (I_Na + I_K + I_l + I_m + I_syn + I) / Cm : volt
    """ + """
    I_Na = g_Na * (m ** 3) * h * (V_Na - V) : amp
    m = a_m / (a_m + b_m) : 1
    a_m = (0.32 * (54 + V/mV)) / 
        (1 - exp(-0.25 * (V/mV + 54))) / ms : Hz
    b_m = (0.28 * (27 + V/mV)) / (exp(0.2 * (V/mV + 27)) - 1) / ms : Hz
    h = clip(1 - 1.25 * n, 0, inf) : 1
    """ + """
    I_K = g_K * n ** 4 * (V_K - V) : amp
    dn/dt = (a_n - (a_n * n)) - b_n * n : 1
    a_n = (0.032 * (52 + V/mV)) / 
        (1 - exp(-0.2 * (V/mV + 52))) / ms : Hz
    b_n = 0.5 * exp(-0.025 * (57 + V/mV)) / ms : Hz
    """ + """
    I_l = g_l * (V_l - V) : amp
    """ + """
    I_m = w_m * w * (V_K - V) : amp
    dw/dt = (w_inf - w) / tau_w/ms : 1
    w_inf = 1 / (1 + exp(-1 * (V/mV + 35) / 10)) : 1
    tau_w = 400 / ((3.3 * exp((V/mV + 35)/20)) + 
        (exp(-1 * (V/mV + 35) / 20))) : 1
    """ + """
    I_syn = g_e * (V_e - V) +
        g_i * (V_i - V) : amp
    g_e : siemens
    g_i : siemens
    """ + """
    I_stim = g_s * (V_e - V) : amp
    g_s : siemens
    """ + """
    I : amp
    """

    syn_e_in = """
    dg/dt = -g / tau_d_ampa : siemens
    g_e_post = g : siemens (summed)
    """

    syn_e_stim = """
    dg/dt = -g / tau_d_ampa : siemens
    g_s_post = g : siemens (summed)
    """

    syn_e = """
    dg/dt = -g  * 1 / (tau_d_ampa - tau_r_ampa): siemens
    g_e_post = g : siemens (summed)
    """

    syn_i = """
    dg/dt = -g * 1 / (tau_d_gaba - tau_r_gaba) : siemens
    g_i_post = g : siemens (summed)
    """

    # --
    # Build networks
    P_i = NeuronGroup(
        N_i,
        model=wb,
        threshold='v >= V_thresh',
        refractory=2 * ms,
        method='exponential_euler'
    )
    P_e = NeuronGroup(
        N_e, model=hh,
        threshold='V >= V_thresh',
        refractory=3 * ms,
        method='exponential_euler'
    )
    P_i.I = I_i
    P_e.I = np.random.uniform(I_e[0], I_e[1], N_e) * uamp
    P_e.V = V_l
    P_i.v = EL_wb

    P_e_back = PoissonGroup(N_e, rates=r_e)

    # --
    # Stimulus
    time_stim = time_stim / second
    window = 500 / 1000.
    t_min = time_stim - window / 2
    t_max = time_stim + window / 2
    stdev = 100 / 1000.  # 100 ms

    # rate = 5  # Hz
    k = N_stim * int(rate_stim / 2)

    ts, idxs = gaussian_impulse(time_stim, t_min,
                                t_max, stdev, N_stim, k,
                                decimals=decimals)
    P_stim = SpikeGeneratorGroup(N_stim, idxs, ts * second)

    # --
    # Connections
    # External
    C_stim_e = Synapses(
        P_stim, P_e, model=syn_e_stim,
        pre='g += w_e', connect='i == j'
    )
    C_back_e = Synapses(
        P_e_back, P_e, model=syn_e_in,
        pre='g += w_e', connect='i == j'
    )

    # Internal
    C_ie = Synapses(P_i, P_e, model=syn_i, pre='g += w_ie', delay=cdelay)
    C_ie.connect(True, p=0.4)
    C_ii = Synapses(P_i, P_i, model=syn_i, pre='g += gSyn_wb', connect=True)

    # --
    # Record
    spikes_i = SpikeMonitor(P_i)
    spikes_e = SpikeMonitor(P_e)
    spikes_stim = SpikeMonitor(P_stim)
    pop_e = PopulationRateMonitor(P_e)
    pop_i = PopulationRateMonitor(P_i)
    voltages_e = StateMonitor(P_e, ('V', 'g_e', 'g_i', 'g_s'), record=range(11, 31))
    voltages_i = StateMonitor(P_i, ('v', 'g_i'), record=range(11, 31))

    # --
    # Go!
    defaultclock.dt = time_step
    run(time, report='text')

    return {
        'spikes_i': spikes_i,
        'spikes_e': spikes_e,
        'spikes_stim': spikes_stim,
        'pop_e': pop_e,
        'pop_i': pop_i,
        'voltages_e': voltages_e,
        'voltages_i': voltages_i
    }


def save_result(name, result, fs=10000):
    spikes_e = result['spikes_e']
    spikes_i = result['spikes_i']
    spikes_stim = result['spikes_i']
    pop_e = result['pop_e']
    pop_i = result['pop_i']
    voltages_e = result['voltages_e']
    voltages_i = result['voltages_i']

    # --
    # Save full
    # Spikes
    np.savetxt(name + '_spiketimes_e.csv',
               np.vstack([spikes_e.i, spikes_e.t, ]).transpose(),
               fmt='%.i, %.5f')
    np.savetxt(name + '_spiketimes_i.csv',
               np.vstack([spikes_i.i, spikes_i.t, ]).transpose(),
               fmt='%.i, %.5f')
    np.savetxt(name + '_spiketimes_stim.csv',
               np.vstack([spikes_stim.i, spikes_stim.t, ]).transpose(),
               fmt='%.i, %.5f')

    # Example trace
    np.savetxt(name + '_exampletrace_e.csv',
               np.vstack([voltages_e.t, voltages_e.V[0], ]).transpose(),
               fmt='%.5f, %.4f')
    np.savetxt(name + '_exampletrace_i.csv',
               np.vstack([voltages_i.t, voltages_i.v[0], ]).transpose(),
               fmt='%.5f, %.4f')

    # Pop rate
    np.savetxt(name + '_poprates_e.csv',
               np.vstack([pop_e.t, pop_e.rate / Hz, ]).transpose(),
               fmt='%.5f, %.1f')
    np.savetxt(name + '_poprates_i.csv',
               np.vstack([pop_i.t, pop_i.rate / Hz, ]).transpose(),
               fmt='%.5f, %.1f')

    # LFP
    lfp = (np.abs(voltages_e.g_e.sum(0)) +
           np.abs(voltages_e.g_i.sum(0)) +
           np.abs(voltages_i.g_i.sum(0)))
    np.savetxt(name + '_lfp.csv', zscore(lfp), fmt='%.2f')

    # PSD
    lfp = lfp[1000:]  # Drop initial spike
    freqs, spec = create_psd(lfp, fs)
    np.savetxt(name + '_psd.csv', np.vstack(
        [freqs, np.log10(spec)]).transpose(), fmt='%.1f, %.3f')

# Using fn in ing.py instead.
# def analyze_result(name, stim, result, fs=10000, save=True):
#     analysis = {}
#
#     spikes_e = result['spikes_e']
#     spikes_stim = result['spikes_stim']
#     spikes_i = result['spikes_i']
#
#     # Get Ns and ts
#     ns_e, ts_e = spikes_e.i, spikes_e.t / second
#     ns_i, ts_i = spikes_i.i, spikes_i.t / second
#     ns_stim, ts_stim = spikes_stim.i, spikes_stim.t / second
#
#     # Drop times before stim time
#     mask = ts_e > stim
#     ns_e, ts_e = ns_e[mask], ts_e[mask]
#
#     # Keep only neurons 0-199
#     mask = ns_e < 200
#     ns_e, ts_e = ns_e[mask], ts_e[mask]
#
#     # kappa
#     r_e = futil.kappa(ns_e, ts_e, ns_e, ts_e, (0, 1), 1.0 / 1000)  # 1 ms bins
#     analysis['kappa_e'] = r_e
#     r_i = futil.kappa(ns_i, ts_i, ns_i, ts_i, (0, 1), 1.0 / 1000)  # 1 ms bins
#     analysis['kappa_i'] = r_i
#
#     # fano
#     fanos_e = futil.fano(ns_e, ts_e)
#     mfano_e = np.nanmean([x for x in fanos_e.values()])
#     analysis['fano_e'] = mfano_e
#
#     # l distance and spike
#     ordered_e, _ = futil.ts_sort(ns_e, ts_e)
#     ordered_stim, _ = futil.ts_sort(ns_stim, ts_stim)
#     lev = futil.levenshtein(list(ordered_stim), list(ordered_e))
#     analysis['lev_e'] = lev
#
#     # ISI and SPIKE
#     sto_e = spk.SpikeTrain(ts_e, (0.5, 1))
#     sto_stim = spk.SpikeTrain(ts_stim, (0.5, 1))
#     sto_e.sort()
#     sto_stim.sort()
#     isi = spk.isi_distance(sto_stim, sto_e)
#     sync = spk.spike_sync(sto_stim, sto_e)
#
#     analysis['isi_e'] = isi
#     analysis['sync_e'] = sync
#
#     if save:
#         with open(name + '_analysis.csv', 'w') as f:
#             [f.write('{0},{1}\n'.format(k, v)) for k, v in analysis.items()]
#
#     return analysis
