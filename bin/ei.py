#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Implements a sparse PING E-I model, based Wang and Buzsaki[0].

[0]: Gamma oscillation by synaptic inhibition in a hippocampal 
interneuronal network model. Wang XJ, Buzsaki G.J Neurosci. 
1996 Oct 15;16(20):6402-13.
"""
from __future__ import division
import argparse
import numpy as np
from brian2 import *
from syncological.ei import model, analyze_result, save_result


parser = argparse.ArgumentParser(
    description="A sparse PING E-I model.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "name",
    help="Name of exp."
)
parser.add_argument(
    "-t", "--time",
    help="Simulation run time (in ms)",
    default=2,
    type=float
)
parser.add_argument(
    "--stim",
    help="Simulus times (in ms)",
    nargs='+',
    default=[1.5],
    type=float
)
parser.add_argument(
    "--rate",
    help="Stimulus firing rate (approx)",
    default=5,
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
    "--w_ei",
    help="Weight I -> E (msiemens)",
    default=1.0,
    type=float
)
parser.add_argument(
    "--w_ie",
    help="Weight I -> E (msiemens)",
    default=0.5,
    type=float
)
parser.add_argument(
    "--w_ee",
    help="Weight E -> E (msiemens)",
    default=0.1,
    type=float
)
parser.add_argument(
    "--w_ii",
    help="Weight I -> I (msiemens)",
    default=0.1,
    type=float
)
parser.add_argument(
    "--I_e",
    help="E population drive",
    default=0.1,
    type=float,
)
parser.add_argument(
    "--I_i",
    help="I population drive",
    default=0.1,
    type=float,
)
parser.add_argument(
    "--I_i_sigma",
    help="I drive variance",
    type=float,
    default=0.01
)
parser.add_argument(
    "--I_e_sigma",
    help="E drive variance",
    type=float,
    default=0.0
)
parser.add_argument(
    "--seed",
    help="Seed value",
    default=None
)
parser.add_argument(
    "--stdp",
    action="store_true"
)
args = parser.parse_args()

try:
    seed = int(args.seed)
except TypeError:
    seed = None

result = model(
    args.time, args.stim, args.rate,
    args.w_e, args.w_i, args.w_ei, args.w_ie, args.w_ee, args.w_ii,
    args.I_e, args.I_i, args.I_i_sigma, args.I_e_sigma,
    args.stdp, seed=seed
)
save_result(args.name, result)
analysis = analyze_result(args.name, args.stim, result, fs=10000, save=True)
