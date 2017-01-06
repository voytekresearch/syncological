#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Implements a sparse E-I model, loosely based Borges et al PNAS 2005.
"""
from __future__ import division
import numpy as np
from brian2 import *
from syncological.inputs import gaussian_impulse, create_ij
from scipy.stats.mstats import zscore
from scipy.spatial.distance import cosine as cosined
from foof.util import create_psd
from fakespikes import util as futil
import pyspike as spk


def model(name,
          time,
          K,
          N_stim,
          ts_stim,
          ns_stim,
          w_e,
          w_i,
          w_ei,
          w_ie,
          w_ee,
          w_ii,
          I_e=0,
          I_i=0,
          I_i_sigma=0,
          I_e_sigma=0,
          N=400,
          scale_noise=4,
          back_rate=12,
          seed=42,
          conn_seed=42,
          verbose=True,
          parallel=False,
          save=True):
    """Model IIIIIEEEEEEE!"""

    if verbose:
        print(name)

    np.random.seed(seed)

    time = time * second
    time_step = 0.01 * ms
    defaultclock.dt = time_step

    # --
    # Model
    # Network
    N_e = int(N * 0.8)
    N_i = int(N * 0.2)

    delay = 2 * ms
    p_ei = 0.1
    p_ie = 0.1
    p_e = 0.1
    p_ii = 0.6

    w_e = w_e * msiemens
    w_i = w_i * msiemens

    w_ei = w_ei * msiemens
    w_ie = w_ie * msiemens
    w_ee = w_ee * msiemens
    w_ii = w_ii * msiemens

    # Should I be norming for net size all. Bio-sense is?
    # w_e = w_e / (p_e * N_e) * msiemens
    # w_i = w_i / (p_e * N_i) * msiemens
    # w_ei = w_ei / (p_ei * N_e) * msiemens  
    # w_ie = w_ie / (p_ie * N_i) * msiemens
    # w_ee = w_ee / (p_e * N_e) * msiemens
    # w_ii = w_ii / (p_ii * N_i) * msiemens

    w_m = 0 * msiemens  # Read ref 47 to get value

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

    # init nets
    P_e = NeuronGroup(N_e,
                      model=hh,
                      threshold='V >= V_thresh',
                      refractory=3 * ms,
                      method='rk2')

    P_i = NeuronGroup(N_i,
                      model=hh,
                      threshold='V >= V_thresh',
                      refractory=3 * ms,
                      method='rk2')

    P_stim = SpikeGeneratorGroup(N_stim, ns_stim, ts_stim * second)

    # init static net params
    P_e.V = V_l
    P_i.V = V_l

    if np.allclose(I_e_sigma, 0.0):
        P_e.I = I_e * uamp
    else:
        P_e.I = np.random.normal(I_e, I_e_sigma, N_e) * uamp

    if np.allclose(I_i_sigma, 0.0):
        P_i.I = I_i * uamp
    else:
        P_i.I = np.random.normal(I_i, I_i_sigma, N_i) * uamp

    # --
    # Syn connections
    # Internal
    conn_prng = np.random.RandomState(int(conn_seed))

    # Internal
    i, j, conn_prng = create_ij(p_ei, len(P_e), len(P_i), conn_prng)
    C_ei = Synapses(P_e, P_i, on_pre='g_e += w_ei', delay=delay)
    C_ei.connect(i=i, j=j)

    i, j, conn_prng = create_ij(p_ie, len(P_i), len(P_e), conn_prng)
    C_ie = Synapses(P_i, P_e, on_pre='g_i += w_ie', delay=delay)
    C_ie.connect(i=i, j=j)

    i, j, conn_prng = create_ij(p_ii, len(P_i), len(P_i), conn_prng)
    C_ii = Synapses(P_i, P_i, on_pre='g_i += w_ii', delay=delay)
    C_ii.connect(i=i, j=j)

    i, j, conn_prng = create_ij(p_e, len(P_e), len(P_e), conn_prng)
    C_ee = Synapses(P_e, P_e, on_pre='g_ee += w_ee', delay=delay)
    C_ee.connect(i=i, j=j)

    # External
    i, j, conn_prng = create_ij(p_e, N_stim, N_stim, conn_prng)
    C_stim_e = Synapses(P_stim, P_e[:N_stim], on_pre='g_s += w_e')
    C_stim_e.connect(i=i, j=j)

    i, j, conn_prng = create_ij(p_e, N_stim, int(N_stim / 4), conn_prng)
    C_stim_i = Synapses(P_stim, P_i[:N_stim], on_pre='g_i += w_i')
    C_stim_i.connect(i=i, j=j)

    Nb = 10000
    P_e_back = PoissonGroup(0.8 * Nb, rates=back_rate * Hz)
    P_i_back = PoissonGroup(0.2 * Nb, rates=back_rate * Hz)

    # Adjusted to these numbers by hand, based on the Vm_hat
    # estimation equation (see results below). This equation 
    # Began with the 0.73 and 3.67 that manuscript.
    # Equation and numbers were taken from p742 of
    # Destexhe, A., Rudolph, M. & Paré, D., 2003. 
    # The high-conductance state of neocortical neurons in vivo. 
    # Nature Reviews Neuroscience, 4(9), pp.739–751.
    w_e_back = scale_noise * 0.73 * g_l
    w_i_back = 2.85 * w_e_back

    p_back = 0.1
    C_back_e = Synapses(P_e_back, P_e, on_pre='g_e += w_e_back')
    C_back_e.connect(True, p=p_back)
    C_back_i = Synapses(P_i_back, P_e, on_pre='g_i += w_i_back')
    C_back_i.connect(True, p=p_back)

    # --
    # Create network
    net = Network(P_e, P_i, P_stim, C_ee, C_ii, C_ie, C_ei, C_stim_e, P_e_back,
                  P_i_back, C_back_e, C_back_i)
    net.store('trials')

    # --
    # Go! Loop over trials (if present)
    results = []
    for k in range(K):
        # --
        trial_name = "{}_k{}".format(name, k)

        if verbose:
            print(">>> Trial {0}".format(k))

        net.restore('trials')

        # Add fresh monitors before run
        spikes_i = SpikeMonitor(P_i)
        spikes_e = SpikeMonitor(P_e)
        spikes_bi = SpikeMonitor(P_i_back)
        spikes_be = SpikeMonitor(P_e_back)
        spikes_stim = SpikeMonitor(P_stim)

        pop_stim = PopulationRateMonitor(P_stim)
        pop_e = PopulationRateMonitor(P_e)
        pop_i = PopulationRateMonitor(P_i)
        traces_e = StateMonitor(P_e, ('V', 'g_e', 'g_i', 'g_s', 'g_ee'),
                                record=True)
        traces_i = StateMonitor(P_i, ('V', 'g_e', 'g_i'), record=True)

        monitors = [
            spikes_i, spikes_e, spikes_be, spikes_bi, spikes_stim, pop_stim,
            pop_e, pop_i, traces_e, traces_i
        ]
        net.add(monitors)

        # --
        report = None
        if verbose:
            report = 'text'
        net.run(time, report=report)

        # --
        # Post-run processing
        # Extract connectivity data
        connected_e = (C_stim_e.i_, C_stim_e.j_)
        connected_i = (C_stim_i.i_, C_stim_i.j_)
        connected_ee = (C_ee.i_, C_ee.j_)
        connected_ie = (C_ie.i_, C_ie.j_)
        connected_ei = (C_ei.i_, C_ei.j_)
        connected_ii = (C_ii.i_, C_ii.j_)

        # Useful for assesing background state
        Mge = traces_e.g_e[:].mean()
        Mgi = traces_e.g_i[:].mean()
        Vm_hat = float((
            (g_l * V_l) + (Mge * V_e) + (Mgi * V_i)) / (g_l + Mge + Mgi))

        Vm = np.mean(traces_e.V_[:])
        Vm_std = np.std(traces_e.V_[:])

        result = {
            'N_stim': N_stim,
            'Vm_hat': Vm_hat,
            'K': K,
            'Vm': Vm,
            'Vm_std': Vm_std,
            'time': time / second,
            'dt': time_step / second,
            'spikes_i': spikes_i,
            'spikes_e': spikes_e,
            'spikes_bi': spikes_bi,
            'spikes_be': spikes_be,
            'spikes_stim': spikes_stim,
            'pop_e': pop_e,
            'pop_stim': pop_stim,
            'pop_i': pop_i,
            'traces_e': traces_e,
            'traces_i': traces_i,
            'connected_e': connected_e,
            'connected_i': connected_i,
            'connected_ie': connected_ie,
            'connected_ei': connected_ei,
            'connected_ii': connected_ii,
            'connected_ee': connected_ee
        }

        # Output
        if save:
            save_result(trial_name, result)
        if parallel:
            result = None

        results.append(result)

        # Reset monitors
        net.remove(monitors)
        net.store('trials')

    # --
    # At the end....
    # Save fixed params
    params = {
        'N_e': N_e,
        'N_i': N_i,
        'I_e': float(I_e),
        'I_i': float(I_i),
        'delay': float(delay),
        'p_ei': float(p_ei),
        'p_ie': float(p_ie),
        'p_e': float(p_e),
        'p_ii': float(p_ii),
        'w_e': float(w_e),
        'w_i': float(w_i),
        'w_ei': float(w_ei),
        'w_ie': float(w_ie),
        'w_ee': float(w_ee),
        'w_ii': float(w_ii),
        'w_m': float(w_m),
        'scale_noise': scale_noise,
        'time': float(time),
        'dt': float(time_step)
    }
    with open(name + '_params.csv', 'w') as f:
        [f.write('{0},{1:.5e}\n'.format(k, v)) for k, v in params.items()]

    return results


def save_result(name, result, fs=10000):
    time = result['time']

    spikes_e = result['spikes_e']
    spikes_i = result['spikes_i']
    spikes_be = result['spikes_be']
    spikes_bi = result['spikes_bi']
    spikes_stim = result['spikes_stim']

    pop_e = result['pop_e']
    pop_i = result['pop_i']
    traces_e = result['traces_e']
    traces_i = result['traces_i']

    i_e, j_e = result['connected_e']
    i_i, j_i = result['connected_i']
    i_ee, j_ee = result['connected_ee']
    i_ei, j_ei = result['connected_ei']
    i_ie, j_ie = result['connected_ie']
    i_ii, j_ii = result['connected_ii']

    ts_e = spikes_e.t_[:]
    ts_stim = spikes_stim.t_[:]
    ns_e = spikes_e.i_[:]
    ns_stim = spikes_stim.i_[:]

    ts_i = spikes_i.t_[:]
    ns_i = spikes_i.i_[:]

    v_e = traces_e.V_[:]
    v_i = traces_i.V_[:]

    # --
    # Spikes
    np.savetxt(name + '_spiketimes_e.csv',
               np.vstack([spikes_e.i,
                          spikes_e.t, ]).transpose(),
               fmt='%i, %.5f')
    np.savetxt(name + '_spiketimes_i.csv',
               np.vstack([spikes_i.i,
                          spikes_i.t, ]).transpose(),
               fmt='%i, %.5f')
    np.savetxt(name + '_spiketimes_back_e.csv',
               np.vstack([spikes_be.i,
                          spikes_be.t, ]).transpose(),
               fmt='%i, %.5f')
    np.savetxt(name + '_spiketimes_back_i.csv',
               np.vstack([spikes_bi.i,
                          spikes_bi.t, ]).transpose(),
               fmt='%i, %.5f')
    np.savetxt(name + '_spiketimes_stim.csv',
               np.vstack([spikes_stim.i,
                          spikes_stim.t, ]).transpose(),
               fmt='%i, %.5f')

    # Example trace
    np.savetxt(name + '_exampletrace_e.csv',
               np.vstack([traces_e.t,
                          traces_e.V[0], ]).transpose(),
               fmt='%.5f, %.4f')
    np.savetxt(name + '_exampletrace_i.csv',
               np.vstack([traces_i.t,
                          traces_i.V[0], ]).transpose(),
               fmt='%.5f, %.4f')

    # Pop rate
    np.savetxt(name + '_poprates_e.csv',
               np.vstack([pop_e.t,
                          pop_e.rate / Hz, ]).transpose(),
               fmt='%.5f, %.1f')
    np.savetxt(name + '_poprates_i.csv',
               np.vstack([pop_i.t,
                          pop_i.rate / Hz, ]).transpose(),
               fmt='%.5f, %.1f')

    # and neuron indices for weight can become a (i, j) matrix
    np.savetxt(name + '_i_e.csv', i_e, fmt='%i')
    np.savetxt(name + '_j_e.csv', j_e, fmt='%i')
    np.savetxt(name + '_i_i.csv', i_i, fmt='%i')
    np.savetxt(name + '_j_i.csv', j_i, fmt='%i')
    np.savetxt(name + '_i_ee.csv', i_ee, fmt='%i')
    np.savetxt(name + '_j_ee.csv', j_ee, fmt='%i')
    np.savetxt(name + '_i_ei.csv', i_ei, fmt='%i')
    np.savetxt(name + '_j_ei.csv', j_ei, fmt='%i')
    np.savetxt(name + '_i_ie.csv', i_ie, fmt='%i')
    np.savetxt(name + '_j_ie.csv', j_ie, fmt='%i')
    np.savetxt(name + '_i_ii.csv', i_ii, fmt='%i')
    np.savetxt(name + '_j_ii.csv', j_ii, fmt='%i')

    # LFP
    lfp = (np.abs(traces_e.g_e.sum(0)) + np.abs(traces_e.g_i.sum(0)) +
           np.abs(traces_e.g_ee.sum(0)))
    np.savetxt(name + '_lfp.csv', zscore(lfp), fmt='%.2f')

    # PSD
    lfp = lfp[:]  # Drop initial spike
    freqs, spec = create_psd(lfp, fs)
    np.savetxt(name + '_psd.csv',
               np.vstack([freqs, np.log10(spec)]).transpose(),
               fmt='%.1f, %.3f')

    # STA
    sta_e, bins_sta_e = futil.spike_triggered_average(
        ts_e, ns_e, v_e, (0, time), 10e-3, 1 / 1e-5)
    np.savetxt(name + '_sta_e.csv',
               np.vstack([sta_e, bins_sta_e]).transpose(),
               fmt='%.5f, %.5f')

    sta_i, bins_sta_i = futil.spike_triggered_average(
        ts_i, ns_i, v_i, (0, time), 10e-3, 1 / 1e-5)
    np.savetxt(name + '_sta_i.csv',
               np.vstack([sta_i, bins_sta_i]).transpose(),
               fmt='%.5f, %.5f')
