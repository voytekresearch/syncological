#!/usr/bin/env python
"""
Wang-Buszaki model
------------------

J Neurosci. 1996 Oct 15;16(20):6402-13.
Gamma oscillation by synaptic inhibition in a hippocampal interneuronal
network model. Wang XJ, Buzsaki G.

Note that implicit integration (exponential Euler) cannot be used,
and therefore simulation is rather slow.
"""
from __future__ import division
import numpy as np
from brian2 import *

np.random.seed()

# User globals
# Neuron #
N_e = 800
N_i = 200

# Run time
time = 1000 * ms
time_step = 0.1 * ms

k_r_e = 1
r_e = k_r_e * 10 * Hz
r_i = r_e

# Network params
# p_syn = 0.4  # if sparse
p_ii = 1.0
p_ie = 0.4
# p_ee = 0.1

# Current inputs
I_currents_wb = 1.1 # * uamp
I_currents_wb = np.random.normal(I_currents_wb, 0.01, N_i) * uamp
I_currents_hh = 0 * uamp

delay = 0 * ms  # Delay between P_e

# After-hyperpolaization (AHP)
# adjust for WB neurons -
# 'theta' in the paper
ahp = 5

# Neuronal params
# HH taken from Figure 3: http://www.ncbi.nlm.nih.gov/pubmed/19011929
# WB taken from: http://www.jneurosci.org/content/16/20/6402.abstract

# Cm_wb = 1 * uF            # /cm**2
area_wb = 1 * cm ** 2
Cm_wb = (1 * ufarad * cm ** -2) * area_wb
Cm = 1 * uF # /cm2

gL_wb = (0.1 * msiemens * cm ** -2) * area_wb
EL_wb = -70 * mV
Vt_wb = -52 * mV
Vt_hh = -63 * mV
ENa_wb = 55 * mV
gNa_wb = (35 * msiemens * cm ** -2) * area_wb
EK_wb = -90 * mV
gK_wb = 9 * msiemens

# New E (HH) cell prop
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

# I-I, taken from
# http://www.ncbi.nlm.nih.gov/pubmed/8815919
ESyn_wb = -75 * mV
gSyn_wb = (0.1 / N_i) * msiemens

# E
taue_hh = 5 * ms
taui_hh = 10 * ms
win_hh = 26 * nsiemens

w_e = 0.06 * msiemens
w_i = 0.02 * msiemens

w_m = 0 / N_e * msiemens # Read ref 47 to get value
w_ei = 1 / N_e * msiemens
w_ee = 0 / N_e * msiemens
w_ie = 0.1 / N_i * msiemens
w_ii = 0.1 / N_i * msiemens
w_m = 0 / N_e * msiemens # Read ref 47 to get value

# Borrowing params from Vogels and Abbot, 2005
# we_hh = (12.1 / (N_e * p_ee)) * nsiemens # (2 /(N_e * p_ee))  * nsiemens
# wi_hh = (40 / (N_i * p_ie)) * nsiemens
# win_hh was tuned by hand until Poisson spike -> HH spike
# the aim was to give the HH net poisson rates
# at the cost of emperical/literature gounding

# http://www.ncbi.nlm.nih.gov/pubmed/16093332
# but we_hh was renormed by avg synapse number
# otherwise the ee driven ringing dominated
# all other dynamics, Poisson and WB.
# Previous sims did not, it seems,
# pick ee and ii 'g' values from the
# emperical lit, so I feel ok taking this
# renorm shortcut ad hoc as it may be.

# we_hh = (1.3 /(N_e * p_ee)) * nsiemens
# wi_hh = (8.75 / (N_i * p_ie)) * nsiemens
# we_hh and wi_hh taken from

# W-B inhibitory neurons
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

# """ + """
#     Isyn =  gSyn_wb_v * (v - ESyn_wb) : amp
#     gSyn_wb_v : siemens
# """ + """

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

V_thresh = 20 * mV
P_i = NeuronGroup(
    N_i,
    model=wb,
    threshold='v >= V_thresh',
    refractory=3*ms,
    method='exponential_euler'
)

P_e = NeuronGroup(
    N_e, model=hh,
    threshold='V >= V_thresh',
    refractory=3*ms,
    method='exponential_euler'
)
# P_e_stim1 = P_e[11:31]
# P_e_stim2 = P_e[71:91]


# Init
P_i.I = I_currents_wb
P_e.I = I_currents_hh
P_e.V = V_l
P_i.v = EL_wb

# --
# Syn
# Ext
P_e_in = PoissonGroup(N_e, rates=r_e)
P_i_in = PoissonGroup(N_i, rates=r_i)
# P_stim1 = PoissonGroup(N_stim, rates=r_stim1)
#
# # --
# # Syn
# # Ext
C_in_e = Synapses(P_e_in, P_e, model=syn_e_in, pre='g += w_e', connect='i == j')
C_in_i = Synapses(P_i_in, P_i, model=syn_e_in, pre='g += w_e/5', connect='i == j')
#
# C_stim1_e = Synapses(P_stim1, P_e, model=syn_e_in, pre='g += w_e_stim')
# C_stim1_e.connect(range(0, 20), range(11, 31))
# C_stim1_i = Synapses(P_stim1, P_i, model=syn_e_in, pre='g += w_e_stim')
# C_stim1_i.connect(range(0, 20), range(11, 31))

# ING
C_ii = Synapses(P_i, P_i, model=syn_i, pre='g += gSyn_wb')
C_ii.connect(True, p=p_ii)
C_ie = Synapses(P_i, P_e, model=syn_i, pre='g += w_ie')
C_ie.connect(True, p=p_ie)
# C_ee_ampa = Synapses(P_e, P_e, model=syn_ampa, pre='g_ampa += g_ee')

# --
# Record
spikes_i = SpikeMonitor(P_i)
spikes_e = SpikeMonitor(P_e)
voltages_e = StateMonitor(P_e, ('V', 'g_e', 'g_i'), record=range(11, 31))
voltages_i = StateMonitor(P_i, ('v', 'g_i'), record=range(11, 31))

defaultclock.dt = time_step
run(time, report='text')

