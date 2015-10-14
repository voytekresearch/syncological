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
from syncological.ping import model, analyze


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
    "--time_stim",
    help="Simulus times (in ms)",
    nargs='+',
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
    "--seed",
    help="Seed value",
    default=None
)
args = parser.parse_args()

# --
# argvs
time = args.time * second
time_stim = args.time_stim * second

# --
# Run!
res = model(time, time_stim, args.w_e, args.w_i, args.w_ei, 
            args.w_ie, seed=args.seed)

# --
# Analysis
# TODO
analyze(res)

