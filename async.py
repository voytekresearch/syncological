#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Implements a sparse balanced and asynchronous E-I model, loosely based 
on Borges and Kopell, 2005.
"""
from __future__ import division
import argparse
import numpy as np
from brian2 import *
from syncological.inputs import gaussian_impulse


def model(time, time_stim, rate_stim, w_e, w_i, w_ei, w_ie, seed=None):
    # def model(time, stim, w_e, w_i, w_ei, w_ie, seed=None):
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

    delay = 2 * ms
    p_ei = 0.4
    p_ie = 0.4
    p_ii = 1.0

    w_e = w_e * msiemens
    w_i = w_i * msiemens
    w_ei = w_ei / (p_ei * N_e) * msiemens
    w_ie = w_ie / (p_ie * N_i) * msiemens

    w_ii = 0.0 / N_i * msiemens
    w_m = 0.0 * msiemens  # Read ref 47 to get value

    # --
    # Model
    np.random.seed(seed)
    time_step = 0.1 * ms
    decimals = 4

    # --
    # cell biophysics
    I_e = [0.01, 0.1]
    I_i = [0.01, 0.1]

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
    P_e = NeuronGroup(
        N_e, model=hh,
        threshold='V >= V_thresh',
        refractory=3 * ms,
        method='exponential_euler'
    )

    P_i = NeuronGroup(
        N_i,
        model=hh,
        threshold='V >= V_thresh',
        refractory=3 * ms,
        method='exponential_euler'
    )
    P_e.V = V_l
    P_i.V = V_l
    P_e.I = np.random.uniform(I_e[0], I_e[1], N_e) * uamp
    P_i.I = np.random.uniform(I_i[0], I_i[1], N_i) * uamp

    P_e_back = PoissonGroup(N_e, rates=r_e)
    P_i_back = PoissonGroup(N_i, rates=r_i)

    # --
    # Stimulus
    time_stim = time_stim / second
    window = 500 / 1000.
    t_min = time_stim - window / 2
    t_max = time_stim + window / 2
    stdev = 100 / 1000.  # 100 ms

    rate = 5  # Hz
    k = N_stim * int(rate / 2)

    ts, idxs = gaussian_impulse(time_stim, t_min, t_max, stdev, N_stim, k,
                                decimals=decimals)
    P_stim = SpikeGeneratorGroup(N_stim, idxs, ts * second)

    # --
    # Connections
    # External
    C_stim_e = Synapses(P_stim, P_e, model=syn_e_stim,
                        pre='g += w_e', connect='i == j')
    C_back_e = Synapses(P_e_back, P_e, model=syn_e_in, pre='g += w_e',
                        connect='i == j')
    C_back_i = Synapses(P_i_back, P_i, model=syn_e_in, pre='g += w_i',
                        connect='i == j')

    # Internal
    C_ei = Synapses(P_e, P_i, model=syn_e, pre='g += w_ei', delay=delay)
    C_ei.connect(True, p=p_ei)

    C_ie = Synapses(P_i, P_e, model=syn_i, pre='g += w_ie', delay=delay)
    C_ie.connect(True, p=p_ie)

    C_ii = Synapses(P_i, P_i, model=syn_i, pre='g += w_ii', delay=delay)
    C_ii.connect(True, p=p_ii)

    # --
    # Record
    spikes_i = SpikeMonitor(P_i)
    spikes_e = SpikeMonitor(P_e)
    spikes_stim = SpikeMonitor(P_stim)
    pop_e = PopulationRateMonitor(P_e)
    pop_i = PopulationRateMonitor(P_i)
    voltages_e = StateMonitor(
        P_e, ('V', 'g_e', 'g_i', 'g_s'), record=range(11, 31))
    voltages_i = StateMonitor(P_i, ('V', 'g_e', 'g_i'), record=range(11, 31))

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
