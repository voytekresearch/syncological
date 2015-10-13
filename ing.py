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


def model(time, time_stim, w_e, w_i, w_ie, seed=42):
    """Model some BRAINS!"""

    # Network
    N = 1000 
    N_e = int(N * 0.8) 
    N_i = int(N * 0.2)
    N_stim = int(N * 0.2)

    r_e = 10 * Hz
    r_i = r_e
    
    w_e = w_e * msiemens
    w_i = w_i * msiemens
    # w_e = w_e * N_e * msiemens
    # w_i = w_i * N_i * msiemens
    w_ie = w_ie / N_i * msiemens
    
    w_ee = 0 / N_e * msiemens
    w_ii = 0.1 / N_i * msiemens
    w_m = 0 / N_e * msiemens # Read ref 47 to get value

    # --
    # Model
    np.random.seed(seed)
    time_step = 0.1 * ms
    decimals = 4
    
    # --
    # cell biophysics   
    # WB (I)
    I_currents_i = 1.1 # * uamp
    I_currents_i = np.random.normal(I_currents_i, 0.01, N_i) * uamp
    I_currents_e = 0.25 * uamp

    # After-hyperpolaization (AHP)
    ahp = 5

    area_wb = 1 * cm ** 2
    Cm_wb = (1 * ufarad * cm ** -2) * area_wb
    Cm = 1 * uF # /cm2

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
    Cm = 1 * uF # /cm2

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
        alpham = (-0.1/mV) * (v + 35*mV) / (exp(-0.1/mV * (v + 35*mV)) - 1)/ms : Hz
        betam  = 4 * exp(-(v + 60*mV) / (18*mV))/ms : Hz
        alphah = 0.07 * exp(-(v + 58*mV) / (20*mV))/ms : Hz
        betah  = 1.0 / (exp(-0.1/mV * (v + 28*mV)) + 1)/ms : Hz
    """ + """
        IK     = gK_wb * n**4 * (v - EK_wb) : amp
        dn/dt  = ahp * (alphan * (1 - n) - (betan * n)) : 1
        alphan = -0.01/mV * (v + 34*mV) / (exp(-0.1/mV * (v + 34*mV)) - 1)/ms : Hz
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
        tau_w = 400 / ((3.3 * exp((V/mV + 35)/20)) + 
                (exp(-1 * (V/mV + 35) / 20))) : 1
    """ + """
        I_syn = g_e * (V_e - V) +
                g_i * (V_i - V) : amp
        g_e : siemens
        g_i : siemens
    """ + """
        I : amp
    """

    syn_e_in = """
        dg/dt = -g / tau_d_ampa : siemens
        g_e_post = g : siemens (summed)
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
        refractory=2*ms,
        method='exponential_euler'
    )
    P_e = NeuronGroup(
        N_e, model=hh,
        threshold='V >= V_thresh',
        refractory=3*ms,
        method='exponential_euler'
    )

    # --
    # Syn
    # P_e_back = PoissonGroup(N_e, rates=r_e)
    # P_i_back = PoissonGroup(N_i, rates=r_i)

    # Stimulus
    time_stim = time_stim/second
    window = 500/1000.
    t_min = time_stim - window / 2
    t_max = time_stim + window / 2
    stdev = 100/1000. # 100 ms

    rate = 5 # Hz
    k = N_stim * int(rate / 2)

    ts, idxs = gaussian_impulse(time_stim, t_min, t_max, stdev, N_stim, k, 
            decimals=decimals)
    P_stim = SpikeGeneratorGroup(N_stim, idxs, ts*second)

    # --
    # Connections
    # External
    C_stim_e = Synapses(P_stim, P_e, model=syn_e_in, 
        pre='g += w_e', connect='i == j')

    # C_back_e = Synapses(P_e_back, P_e, model=syn_e_in, 
    #     pre='g += w_e', connect='i == j')
    # C_back_i = Synapses(P_i_back, P_i, model=syn_e_in,
    #     pre='g += w_i', connect='i == j')

    # Internal
    C_ie = Synapses(P_i, P_e, model=syn_i, pre='g += w_ie')
    C_ie.connect(True, p=0.4)
    
    C_ii = Synapses(P_i, P_i, model=syn_i, 
                    pre='g += gSyn_wb', connect=True)
    
    # --
    # Init
    P_i.I = I_currents_i
    P_e.I = I_currents_e
    P_e.V = V_l
    P_i.v = EL_wb

    # --
    # Record
    spikes_i = SpikeMonitor(P_i)
    spikes_e = SpikeMonitor(P_e)
    pop_e = PopulationRateMonitor(P_e)
    voltages_e = StateMonitor(P_e, ('V', 'g_e', 'g_i'), record=range(11, 31))
    voltages_i = StateMonitor(P_i, ('v', 'g_i'), record=range(11, 31))

    # --
    # Go!
    defaultclock.dt = time_step
    run(time, report='text')

    return {
        'spikes_i' : spikes_i,
        'spikes_e' : spikes_e,
        'pop_e' : pop_e,
        'voltages_e' : voltages_e,
        'voltages_i' : voltages_i
    }
    
    
def analyze(result):
    pass
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A sparse ING E-I model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "name",
        help="Name of exp, used to save results as hdf5."
    )
    parser.add_argument(
        "-t", "--time",
        help="Simulation run time (in ms)",
        default=2,
        type=float
    )
    parser.add_argument(
        "--time_stim",
        help="Simulus times (in ms)",
        nargs = '+',
        default=[1.5],
        type=float
    )
    parser.add_argument(
        "--w_e",
        help="Input weight to E (msiemens)",
        default=0.06,
        type=float
    )
    parser.add_argument(
        "--w_i",
        help="Input weight to E (msiemens)",
        default=0.02,
        type=float
    )
    parser.add_argument(
        "--w_ie",
        help="Weight I -> E (msiemens)",
        default=0.1,
        type=float
    )
    args = parser.parse_args()

    # --
    # argvs 
    time = args.time * second
    time_stim = args.time_stim * second

    w_e = args.w_e 
    w_i = args.w_i
    w_ie = args.w_ie 

    # --
    # Run!
    res = model(time, time_stim, w_e, w_i, w_ie)

    # -- 
    # Analysis
    # TODO
    analyze(res)
