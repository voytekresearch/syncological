#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Implements a sparse PING E-I model, based Borges et al PNAS 2005.
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


def model(name, time,
          N_stim, ts_stim, ns_stim,
          w_e, w_i, w_ei, w_ie, w_ee, w_ii, I_e, I_i,
          I_i_sigma=0, I_e_sigma=0, N=400, stdp=False, balanced=True,
          seed=None, verbose=True, parallel=False, 
          conn_seed=None, analysis=True, save=True):
    """Model IIIIIEEEEEEE!"""

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

    P_e = NeuronGroup(
        N_e, model=hh,
        threshold='V >= V_thresh',
        refractory=3 * ms,
        method='rk2'
    )

    P_i = NeuronGroup(
        N_i,
        model=hh,
        threshold='V >= V_thresh',
        refractory=3 * ms,
        method='rk2'
    )

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
    P_stim = SpikeGeneratorGroup(N_stim, ns_stim, ts_stim * second)
    
    # --
    # Syn
    # Internal
    if conn_seed:
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
        i, j, conn_prng = create_ij(p_e, len(P_stim), len(P_e), conn_prng)
        C_stim_e = Synapses(P_stim, P_e, on_pre='g_s += w_e')
        C_stim_e.connect(i=i, j=j)

        i, j, conn_prng = create_ij(p_e, len(P_stim) / 4, len(P_i), conn_prng)
        C_stim_i = Synapses(P_stim, P_i, on_pre='g_i += w_i')
        C_stim_i.connect(i=i, j=j)
    else:
        # Internal
        C_ei = Synapses(P_e, P_i, on_pre='g_e += w_ei', delay=delay)
        C_ei.connect(True, p=p_ei)

        C_ie = Synapses(P_i, P_e, on_pre='g_i += w_ie', delay=delay)
        C_ie.connect(True, p=p_ie)

        C_ii = Synapses(P_i, P_i, on_pre='g_i += w_ii', delay=delay)
        C_ii.connect(True, p=p_ii)

        C_ee = Synapses(P_e, P_e, on_pre='g_ee += w_ee', delay=delay)
        C_ee.connect(True, p=p_e)

        # External
        C_stim_e = Synapses(P_stim, P_e[:N_stim], on_pre='g_s += w_e')
        C_stim_e.connect(True, p=p_e)
        C_stim_i = Synapses(P_stim, P_i[:int(N_stim / 4)], on_pre='g_i += w_i')
        C_stim_i.connect(True, p=p_e)

    if balanced:
        Nb = 10000
        P_e_back = PoissonGroup(0.8 * Nb, rates=12 * Hz)  
        P_i_back = PoissonGroup(0.2 * Nb, rates=12 * Hz)

        # Adjusted to these numbers by hand, based on the Vm_hat
        # estimation equation (see results below). This equation 
        # was taken from p742 of
        # Destexhe, A., Rudolph, M. & Paré, D., 2003. 
        # The high-conductance state of neocortical neurons in vivo. 
        # Nature Reviews Neuroscience, 4(9), pp.739–751. 
        # begining with the 0.73 and 3.67 numbers in that
        # manuscript
        w_e_back = 4 * 0.73 * g_l  
        w_i_back = 2.85 * w_e_back

        p_back = 0.1
        C_back_e = Synapses(P_e_back, P_e, on_pre='g_e += w_e_back')
        C_back_e.connect(True, p=p_back)
        C_back_i = Synapses(P_i_back, P_e, on_pre='g_i += w_i_back')
        C_back_i.connect(True, p=p_back)

    # Learn?
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
            on_pre="""
            g_e += w_stdp
            Apre += dpre_e
            w_stdp = clip(w_stdp + dpost_e, 0, gmax_e)
            """,
            on_post="""
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
            on_pre="""
            g_ee += w_stdp
            Apre += dpre_ee
            w_stdp = clip(w_stdp + dpost_ee, 0, gmax_ee)
            """,
            on_post="""
            Apost += dpost_ee
            w_stdp = clip(w_stdp + dpre_ee, 0, gmax_ee)
            """
        )
        C_ee.connect(True, p=p_e)
        C_ee.w_stdp = 'w_ee + (randn() * 0.1 * w_ee)'

    # --
    # Create network 
    net = Network(
        P_e, P_i, P_stim, C_ee, C_ii, C_ie, C_ei, C_stim_e
    )
    if balanced:
        net.add([P_e_back, P_i_back, C_back_e, C_back_i])

    # Add fresh monitors before run
    spikes_i = SpikeMonitor(P_i)
    spikes_e = SpikeMonitor(P_e)
    spikes_stim = SpikeMonitor(P_stim)
    pop_stim = PopulationRateMonitor(P_stim)
    pop_e = PopulationRateMonitor(P_e)
    pop_i = PopulationRateMonitor(P_i)
    traces_e = StateMonitor(P_e, 
                            ('V', 'g_e', 'g_i', 'g_s', 'g_ee'),
                            record=True)
    traces_i = StateMonitor(P_i, 
                            ('V', 'g_e', 'g_i'),
                            record=True)

    monitors = [spikes_i, spikes_e, spikes_stim,
                pop_stim, pop_e, pop_i, traces_e, traces_i]

    if stdp:
        weights_ee = StateMonitor(C_ee, 'w_stdp', record=True)
        weights_e = StateMonitor(C_stim_e, 'w_stdp', record=True)
        monitors += [weights_e, weights_ee]
    else:
        weights_ee = None
        weights_e = None

    net.add(monitors)

    # -- 
    # Run
    if verbose: print(">>> Running")
    report = None
    if verbose: report = 'text'

    net.run(time, report=report)

    # Post-run processing
    if verbose: print(">>> Analyzing and saving")

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
    Vm_hat = float(((g_l * V_l) + (Mge * V_e) + (Mgi * V_i)) / (g_l + Mge + Mgi))

    Vm = np.mean(traces_e.V_[:])
    Vm_std = np.std(traces_e.V_[:])

    result = {
        'N_stim': N_stim,
        'Vm_hat' : Vm_hat,
        'Vm' : Vm,
        'Vm_std' : Vm_std,
        'time': time / second,
        'dt': time_step / second,
        'spikes_i': spikes_i,
        'spikes_e': spikes_e,
        'spikes_stim': spikes_stim,
        'pop_e': pop_e,
        'pop_stim': pop_stim,
        'pop_i': pop_i,
        'traces_e': traces_e,
        'traces_i': traces_i,
        'weights_e': weights_e,
        'weights_ee': weights_ee,
        'connected_e': connected_e,
        'connected_i': connected_i,
        'connected_ie': connected_ie,
        'connected_ei': connected_ei,
        'connected_ii': connected_ii,
        'connected_ee': connected_ee
    }
    
    if save:
        save_result(name, result)

        # Save params
        params = {
            'N_e' : N_e,
            'N_i' : N_i,
            'I_e' : float(I_e),
            'I_i' : float(I_i),
            'delay' : float(delay),
            'p_ei' : float(p_ei),
            'p_ie' : float(p_ie),
            'p_e' : float(p_e),
            'p_ii' : float(p_ii),
            'w_e' : float(w_e),
            'w_i' : float(w_i),
            'w_ei' : float(w_ei),
            'w_ie' : float(w_ie),
            'w_ee' : float(w_ee),
            'w_ii' : float(w_ii),
            'w_m' : float(w_m),
            'time' : float(time),
            'dt' : float(time_step)
        }
        with open(name + '_params.csv', 'w') as f:
            [f.write('{0},{1:.5e}\n'.format(k, v))
             for k, v in params.items()]

    if analysis:
        analyze_result(name, result, fs=1 / result['dt'], save=True)

    # -- If running in parallel don't return anything to save memory.
    if parallel:
        result = None

    return result


def min_syn(connected_i, connected_j, n_syn):
    """Find all post-synaptic neurons (j) with a minimum number
    of synapses, and all the pre-synaptic neurons that make those
    connections.

    Params
    -----
    connected_i : sequence
        List of pre-synaptic neurons
    connected_j : sequence
        List of post-synaptic neurons
    n_syn : int
        Minimum synpase number
    """
    # Find neurons who have >= n_syn
    sel_i, sel_j = [], []
    for i, j in zip(connected_i, connected_j):
        if np.sum(j == connected_j) >= n_syn:
            sel_i.append(i)
            sel_j.append(j)

    ns_i = np.unique(sel_i)
    ns_j = np.unique(sel_j)
    
    return ns_i, ns_j


def save_result(name, result, fs=10000):
    time = result['time']

    spikes_e = result['spikes_e']
    spikes_i = result['spikes_i']
    spikes_stim = result['spikes_i']
    pop_e = result['pop_e']
    pop_i = result['pop_i']
    traces_e = result['traces_e']
    traces_i = result['traces_i']
    weights_e = result['weights_e']
    weights_ee = result['weights_ee']

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
               np.vstack([spikes_e.i, spikes_e.t, ]).transpose(),
               fmt='%i, %.5f')
    np.savetxt(name + '_spiketimes_i.csv',
               np.vstack([spikes_i.i, spikes_i.t, ]).transpose(),
               fmt='%i, %.5f')
    np.savetxt(name + '_spiketimes_stim.csv',
               np.vstack([spikes_stim.i, spikes_stim.t, ]).transpose(),
               fmt='%i, %.5f')

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

    # Save first and last weights
    if weights_e is not None:
        np.savetxt(name + '_w_e.csv',
                   np.vstack([weights_e.w_stdp_[:, 0],
                              weights_e.w_stdp_[:, weights_e.w_stdp_.shape[1] - 1]]),
                   fmt='%.8f')

    if weights_ee is not None:
        np.savetxt(name + '_w_ee.csv',
                   np.vstack([weights_ee.w_stdp_[:, 0],
                              weights_ee.w_stdp_[:, weights_ee.w_stdp_.shape[1] - 1]]),
                   fmt='%.8f')

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
    lfp = (np.abs(traces_e.g_e.sum(0)) +
           np.abs(traces_e.g_i.sum(0)) +
           np.abs(traces_e.g_ee.sum(0)))
    np.savetxt(name + '_lfp.csv', zscore(lfp), fmt='%.2f')

    # PSD
    lfp = lfp[:]  # Drop initial spike
    freqs, spec = create_psd(lfp, fs)
    np.savetxt(name + '_psd.csv', np.vstack(
        [freqs, np.log10(spec)]).transpose(), fmt='%.1f, %.3f')

    # STA
    sta_e, bins_sta_e = futil.spike_triggered_average(
        ts_e, ns_e, v_e, (0, time), 10e-3, 1 / 1e-5)
    np.savetxt(name + '_sta_e.csv', np.vstack(
        [sta_e, bins_sta_e]).transpose(), fmt='%.5f, %.5f')

    sta_i, bins_sta_i = futil.spike_triggered_average(
        ts_i, ns_i, v_i, (0, time), 10e-3, 1 / 1e-5)
    np.savetxt(name + '_sta_i.csv', np.vstack(
        [sta_i, bins_sta_i]).transpose(), fmt='%.5f, %.5f')


def analyze_result(name, result, fs=100000, save=True, drop_before=0.1):
    analysis = {}

    # -- Unpack data
    time = result['time']

    spikes_e = result['spikes_e']
    spikes_stim = result['spikes_stim']

    spikes_i = result['spikes_i']
    traces_e = result['traces_e']
    traces_e = result['traces_e']
    traces_i = result['traces_i']

    ts_e = spikes_e.t_[:]
    ts_stim = spikes_stim.t_[:]
    ns_e = spikes_e.i_[:]
    ns_stim = spikes_stim.i_[:]

    ts_i = spikes_i.t_[:]
    ns_i = spikes_i.i_[:]

    v_e = traces_e.V_[:]
    v_i = traces_i.V_[:]
    # -------------------------------------------------------------------------
    # -- Select data
    # Analyze only N_stim
    N_stim = result['N_stim']
    mask = ns_e <= N_stim
    ns_e, ts_e = ns_e[mask], ts_e[mask]
    mask = ns_i <= int(N_stim / 4)
    ns_i, ts_i = ns_i[mask], ts_i[mask]

    # Drop before drop_before
    mask = ts_e >= drop_before
    ns_e, ts_e = ns_e[mask], ts_e[mask]

    mask = ts_i >= drop_before
    ns_i, ts_i = ns_i[mask], ts_i[mask]

    mask = ts_stim >= drop_before
    ns_stim, ts_stim = ns_stim[mask], ts_stim[mask]

    assert ns_e.shape == ts_e.shape
    assert ns_stim.shape == ts_stim.shape

    # -------------------------------------------------------------------------
    # -- Analyze
    # All N
    # Mean rate
    analysis['rate_stim'] = ts_stim.size / time
    analysis['rate_e'] = ts_e.size / time
    analysis['rate_i'] = ts_i.size / time

    # kappa
    r_e = futil.kappa(ns_e, ts_e, ns_e, ts_e, (0, time), 1.0 / 1000)  # 1 ms bins
    analysis['kappa_e'] = r_e
    r_i = futil.kappa(ns_i, ts_i, ns_i, ts_i, (0, time), 1.0 / 1000)  # 1 ms bins
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
    analysis['s_isi_e'] = spk.isi_distance(sto_stim, sto_e)
    analysis['s_sync_e'] = spk.spike_sync(sto_stim, sto_e)

    # lev and KL distance
    ordered_e, _ = futil.ts_sort(ns_e, ts_e)
    ordered_stim, _ = futil.ts_sort(ns_stim, ts_stim)
    analysis['lev_spike_e'] = futil.levenshtein(
        list(ordered_stim), list(ordered_e))
    analysis['lev_spike_e_n'] = futil.levenshtein(
        list(ordered_stim), list(ordered_e)) / len(ordered_stim)

    analysis['kl_spike_e'] = futil.kl_divergence(ordered_stim, ordered_e)

    ra_e, _, _ = futil.rate_code(ts_e, (0, time), 20e-3)
    ra_stim, _, _ = futil.rate_code(ts_stim, (0, time), 20e-3)
    analysis['lev_fine_rate_e'] = futil.levenshtein(ra_stim, ra_e)
    analysis['lev_fine_rate_e_n'] = futil.levenshtein(ra_stim, ra_e) / len(ra_stim)
    analysis['kl_fine_rate_e'] = futil.kl_divergence(ra_stim, ra_e)

    ra_e, _, _ = futil.rate_code(ts_e, (0, time), 50e-3)
    ra_stim, _, _ = futil.rate_code(ts_stim, (0, time), 50e-3)
    analysis['lev_course_rate_e'] = futil.levenshtein(ra_stim, ra_e)
    analysis['lev_course_rate_e_n'] = futil.levenshtein(ra_stim, ra_e) / len(ra_stim)
    analysis['kl_course_rate_e'] = futil.kl_divergence(ra_stim, ra_e)

    # tol = 1e-2  # 10 ms
    # cc_e, _, _ = futil.coincidence_code(ts_e, ns_e, tol)
    # cc_stim, _, _ = futil.coincidence_code(ts_stim, ns_stim, tol)
    # analysis['lev_cc_e'] = futil.levenshtein(cc_stim, cc_e)
    # analysis['kl_cc_e'] = futil.kl_divergence(cc_stim, cc_e)

    # Gamma power
    lfp = (np.abs(traces_e.g_e.sum(0)) +
           np.abs(traces_e.g_i.sum(0)) +
           np.abs(traces_e.g_ee.sum(0)))
    fs, spec = futil.create_psd(lfp, 1 / result['dt'])
    m = np.logical_and(fs >= 20, fs <= 80)
    analysis['pow_mean'] = np.mean(spec[m])
    analysis['pow_std'] = np.std(spec[m])

    # STA distance
    sta_e, _ = futil.spike_triggered_average(
        ts_e, ns_e, v_e, (0, time), 10e-3, 1 / 1e-5)
    sta_i, _ = futil.spike_triggered_average(
        ts_i, ns_i, v_i, (0, time), 10e-3, 1 / 1e-5)
    analysis['sta_distance'] = cosined(sta_e, sta_i)
    analysis['rmse'] = np.sqrt(np.mean((sta_e - sta_i) ** 2))

    # -------------------------------------------------------------------------
    # Only include neurons with at least 3 
    # post-syn connections. The rest are not going 
    # be passing stim's message anyway, probably.
    n_syn = 10
    i_e, j_e = result['connected_e']
    _, ns_j = min_syn(i_e, j_e, n_syn)

    # drop all neurons not in ns_j from both
    # _stim and _e
    m_stim = np.zeros_like(ns_stim, dtype=np.bool)
    m_e = np.zeros_like(ns_e, dtype=np.bool)
    for n in ns_j:
        m_e = np.logical_or(n == ns_e, m_e)
        m_stim = np.logical_or(n == ns_stim, m_stim)

    ns_e, ts_e = ns_e[m_e], ts_e[m_e]
    ns_stim, ts_stim = ns_stim[m_stim], ts_stim[m_stim]

    assert ns_e.size == np.sum(m_e)
    assert ns_e.shape == ts_e.shape
    assert ns_stim.shape == ts_stim.shape

    # Quick estimate of how many spikes are left
    analysis['rate_stim_ms'] = ts_stim.size / time
    analysis['rate_e_ms'] = ts_e.size / time

    # ISI and SPIKE
    sto_e = spk.SpikeTrain(ts_e, (ts_e.min(), ts_e.max()))
    sto_stim = spk.SpikeTrain(ts_stim, (ts_stim.min(), ts_stim.max()))
    sto_e.sort()
    sto_stim.sort()
    analysis['s_isi_ms'] = spk.isi_distance(sto_stim, sto_e)
    analysis['s_sync_ms'] = spk.spike_sync(sto_stim, sto_e)

    # lev and KL distance
    ordered_e, _ = futil.ts_sort(ns_e, ts_e)
    ordered_stim, _ = futil.ts_sort(ns_stim, ts_stim)
    analysis['lev_spike_ms'] = futil.levenshtein(
        list(ordered_stim), list(ordered_e))
    analysis['lev_spike_ms_n'] = futil.levenshtein(
        list(ordered_stim), list(ordered_e)) / len(ordered_stim)
    analysis['kl_spike_ms'] = futil.kl_divergence(ordered_stim, ordered_e)

    ra_e, _, _ = futil.rate_code(ts_e, (0, time), 20e-3)
    ra_stim, _, _ = futil.rate_code(ts_stim, (0, time), 20e-3)
    analysis['lev_fine_rate_ms'] = futil.levenshtein(ra_stim, ra_e)
    analysis['lev_fine_rate_ms_n'] = futil.levenshtein(ra_stim, ra_e) / len(ra_stim)
    analysis['kl_fine_rate_ms'] = futil.kl_divergence(ra_stim, ra_e)

    ra_e, _, _ = futil.rate_code(ts_e, (0, time), 50e-3)
    ra_stim, _, _ = futil.rate_code(ts_stim, (0, time), 50e-3)
    analysis['lev_course_rate_ms'] = futil.levenshtein(ra_stim, ra_e)
    analysis['lev_course_rate_ms'] = futil.levenshtein(ra_stim, ra_e) / len(ra_stim)
    analysis['kl_course_rate_ms'] = futil.kl_divergence(ra_stim, ra_e)

    if save:
        with open(name + '_analysis.csv', 'w') as f:
            [f.write('{0},{1:.3e}\n'.format(k, v))
             for k, v in analysis.items()]

    return analysis
