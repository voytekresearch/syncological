#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Implements a sparse balanced and asynchronous E-I model, loosely based 
on Borges and Kopell, 2005.
"""
from __future__ import division
import argparse
import numpy as np
from brian2 import *
from syncological.async import model
from syncological.ping import analyze_result, save_result


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
    "--stim",
    help="Simulus time (in ms)",
    default=1.5,
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
    default=0.5,
    type=float
)
parser.add_argument(
    "--w_i",
    help="Input weight to E (msiemens)",
    default=0.5,
    type=float
)
parser.add_argument(
    "--w_ei",
    help="Weight E -> I (msiemens)",
    default=0.1,
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

try:
    seed = int(args.seed)
except TypeError:
    seed = None

result = model(
    args.time,
    args.stim, args.rate,
    args.w_e, args.w_i, args.w_ei, args.w_ie,
    seed=seed
)
save_result(args.name, result)
analysis = analyze_result(args.name, args.stim, result, fs=10000, save=True)
