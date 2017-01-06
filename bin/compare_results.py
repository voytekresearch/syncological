#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Usage: compare_results2.py PATH NAME -r REF MODELS...
   [--n_syn=NSYN]
   [--ref_path=RPATH]

Compare model results of a reference model.

Note: assumes models run were from ei2, found in 
`syncological/bin`

    Arguments:
        PATH       path to the data
        NAME       name to give this comparison
        -r REF       reference model number
        MODELS...  model's numbers

    Options:
        -h --help                   show this screen
        --n_syn=NSYN                minimum post-synapse number [default: 0]
        --ref_path=RPATH            optiona seperate path to reference data

"""
from __future__ import division

import numpy as np
import os, sys
import pandas as pd

import syncological as sync
from syncological import ei2
from syncological.results import compare
from fakespikes import neurons, util, rates
from docopt import docopt
from joblib import Parallel, delayed


if __name__ == "__main__":
    args = docopt(__doc__, version='1.0')
    
    path = args['PATH']
    name = args['NAME']

    ref = int(args['-r'])
    models = [int(f) for f in args['MODELS']]
    n_syn = int(args['--n_syn'])

    ref_path = args['--ref_path']
    if ref_path is None:
        ref_path = path

    analysis, skipped = compare(ref, models, path, ref_path=ref_path)
    print("The following could not be compared:\n{}".format(skipped))

    df = pd.DataFrame(analysis)
    df.to_csv(name, sep=",", index=False)
    
