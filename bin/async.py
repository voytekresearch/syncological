#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Implements a sparse balanced and asynchronous E-I model, loosely based 
on Borges and Kopell, 2005.
"""
from __future__ import division
import argparse
import numpy as np
from brian2 import *
from syncological.async import model, analyze


parser = argparse.ArgumentParser(
    description="A sparse, balanced, and asynchronous E-I model.",
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
    help="Simulus time (in ms)",
    default=1.5,
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
    default=0.24,
    type=float
)
parser.add_argument(
    "--w_ei",
    help="Weight E -> I (msiemens)",
    default=0.5,
    type=float
)
parser.add_argument(
    "--w_ie",
    help="Weight I -> E (msiemens)",
    default=0.5,
    type=float
)
args = parser.parse_args()

# --
# argvs 
time = args.time * second
time_stim = args.time_stim * second

w_e = args.w_e 
w_i = args.w_i
w_ei = args.w_ei 
w_ie = args.w_ie 

# --
# Run!
res = model(time, time_stim, w_e, w_i, w_ei, w_ie)

# -- 
# Analysis
# TODO
analyze(res)

