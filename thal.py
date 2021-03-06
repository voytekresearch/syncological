#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Thal input (weak and highly sync) drives gamma [0].

E/I balanced. Dropped I -> I

[0]: https://www.sciencemag.org/content/312/5780/1622.full.pdf
"""
from __future__ import division
import numpy as np
from brian2 import *
from syncological.util import thal_spikes


# --
# Free
time = 1 * second
time_step = 0.1 * ms

N_e = 160
N_i = 40
N_stim = 20

f_thal = 45
N_thal = N_e
k_thal = 20

k_r_e = 1
r_e = k_r_e * 10 * Hz
r_i = r_e

k_stim1 = 0
r_stim1 = k_stim1 * r_e

k_I = 0
I_e_range = (0.7, 0.9)
I_i_range = (0.5, 0.7)

w_e = 0.06 * msiemens
w_i = 0.02 * msiemens
w_e_stim = w_e * N_stim

w_ei = 1 / N_e * msiemens
w_ee = 0 / N_e * msiemens
w_ie = 0.2 / N_i * msiemens
w_ii = 0.1 / N_i * msiemens
w_m = 0 / N_e * msiemens # Read ref 47 to get value

# --
# Fixed
# cell
Cm = 1 * uF # /cm2

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

V_i = -80 * mV
V_e = 0 * mV

# TODO add xi term for indep private noise.
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

# Assumes 1 mM Mg2++; See Kopell, PNAS, 2005.
syn_nmda = """
    ds_nmda/dt = ((1 + tanh(V_pre / 10)) *
                ((1 - s_nmda) / tau_r_nmda) -
                (s_nmda / tau_d_nmda)) *
                (1 / (1 + 3.57 * exp(-0.063 * V_post)) : Hz
"""

# --
# Networks
V_thresh = 20 * mV
P_e = NeuronGroup(
    N_e, model=hh,
    threshold='V >= V_thresh',
    refractory=3*ms
    # method='exponential_euler'
)

P_i = NeuronGroup(
    N_i,
    model=hh,
    threshold='V >= V_thresh',
    refractory=1*ms
    # method='exponential_euler'
)
P_e_in = PoissonGroup(N_e, rates=r_e)
P_i_in = PoissonGroup(N_i, rates=r_i)
P_stim1 = PoissonGroup(N_stim, rates=r_stim1)

times, indices = thal_spikes(time/second, f_thal, N_thal, k_thal)
P_thal = SpikeGeneratorGroup(N_thal, indices, times*second)

# --
# Syn
# Ext
# C_in_e = Synapses(P_e_in, P_e, model=syn_e_in, pre='g += w_e', connect='i == j')
# C_in_i = Synapses(P_i_in, P_i, model=syn_e_in, pre='g += w_i', connect='i == j')

C_thal_e = Synapses(P_thal, P_e, model=syn_e_in, pre='g += w_e', connect='i == j')
# C_thal_i = Synapses(P_thal, P_i, model=syn_e_in, pre='g += w_e', connect='i == j')

# Balance E/I
C_ei = Synapses(P_e, P_i, model=syn_e, pre='g += w_ei')
C_ei.connect(True, p=0.4)

C_ie = Synapses(P_i, P_e, model=syn_i, pre='g += w_ie')
C_ie.connect(True, p=0.4)

C_ii = Synapses(P_i, P_i, model=syn_i, pre='g += w_ii')
C_ii.connect(True, p=1.0)

# C_ei = Synapses(P_e, P_i, model=syn_e, pre='g += w_ei', connect=True)
# C_ie = Synapses(P_i, P_e, model=syn_i, pre='g += w_ie', connect=True)
# C_ii = Synapses(P_i, P_i, model=syn_i, pre='g += w_ii', connect=True)
# C_ee_ampa = Synapses(P_e, P_e, model=syn_ampa, pre='g_ampa += g_ee')
# C_ee_nmda = Synapses(P_e, P_e, model=syn_nmda, pre='g_nmda += g_ee')

# --
# Init
P_e.V = V_l
P_i.V = V_l
P_e.I = np.random.uniform(I_e_range[0], I_e_range[1], N_e) * k_I * uamp
P_i.I = np.random.uniform(I_i_range[0], I_i_range[1], N_i) * k_I * uamp

# --
# Record
spikes_i = SpikeMonitor(P_i)
spikes_e = SpikeMonitor(P_e)
voltages_e = StateMonitor(P_e, ('V', 'g_e', 'g_i'), record=range(11, 31))
voltages_i = StateMonitor(P_i, ('V', 'g_e', 'g_i'), record=range(11, 31))

defaultclock.dt = time_step
run(time, report='text')
