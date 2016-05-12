from __future__ import division

import numpy as np
import os, sys
from collections import defaultdict
from pyspike import SpikeTrain, isi_distance, spike_sync
from fakespikes.util import (kappa, ts_sort, levenshtein, 
                             kl_divergence, rate_code)
from syncological.util import progressbar


def load_results(number, path):
    to_load = [ 
        "exampletrace_i", 
        "i_ee", 
        "i_i", 
        "i_ii", 
        "j_ee", 
        "j_i", 
        "j_ii", 
        "poprates_i", 
        "spiketimes_e", 
        "spiketimes_stim", 
        "sta_i",
        "exampletrace_e",
        "i_e",
        "i_ei",
        "i_ie",
        "j_e",
        "j_ei",
        "j_ie",
        "lfp",
        "poprates_e",
        "psd",
        "spiketimes_i",
        "sta_e"
    ]
    
    result = {}
    for l in to_load:
        f = os.path.join(path, str(number) + "_" + l + ".csv")
        result[l] = np.loadtxt(f, delimiter=",")

    return result


def compare(ref, models, path, n_syn=0):
    """Compate models to a reference model"""

    if n_syn != 0:
        raise NotImplementedError("TODO")

    # --
    # load ref
    ref_data = load_results(ref, path)

    # unpack into ts_ ns_ time etc
    ns_ref = ref_data['spiketimes_e'][:, 0]
    ts_ref = ref_data['spiketimes_e'][:, 1]
    time = ref_data['spiketimes_stim'][:, 1].max()

    # --
    analysis = defaultdict(list)
    for m in progressbar(models):
        if ref == m:
            continue
        analysis['models'].append(m)

        model_data = load_results(m, path)
        ns_e = model_data['spiketimes_e'][:,0]
        ts_e = model_data['spiketimes_e'][:,1]
        
        # --
        # sync
        sto_e = SpikeTrain(ts_e, (ts_e.min(), ts_e.max()))
        sto_ref = SpikeTrain(ts_ref, (ts_ref.min(), ts_ref.max()))
        sto_e.sort()
        sto_ref.sort()

        analysis['s_isi_e'].append(isi_distance(sto_ref, sto_e))
        analysis['s_sync_e'].append(spike_sync(sto_ref, sto_e))

        # --
        # lev and KL distance
        # spike timing code
        ordered_e, _ = ts_sort(ns_e, ts_e)
        ordered_ref, _ = ts_sort(ns_ref, ts_ref)

        analysis['lev_spike_e'].append(
            levenshtein(
                list(ordered_ref), 
                list(ordered_e))
        )
        analysis['lev_spike_e_n'].append(
            levenshtein(
                list(ordered_ref), 
                list(ordered_e)) / len(ordered_ref)
        )
        analysis['kl_spike_e'].append(
            kl_divergence(ordered_ref, ordered_e)
        )
        
        # 'fine rate', AMPA time scale
        ra_e, _, _ = rate_code(ts_e, (0, time), 5e-3)
        ra_ref, _, _ = rate_code(ts_ref, (0, time), 5e-3)

        analysis['lev_fine_rate_e'].append(
            levenshtein(ra_ref, ra_e)
        )
        analysis['lev_fine_rate_e_n'].append(
            levenshtein(ra_ref, ra_e) / len(ra_ref)
        )
        analysis['kl_fine_rate_e'].append(
            kl_divergence(ra_ref, ra_e)
        )
        
        # 'course rate', NMDA time scale
        ra_e, _, _ = rate_code(ts_e, (0, time), 50e-3)
        ra_ref, _, _ = rate_code(ts_ref, (0, time), 50e-3)

        analysis['lev_course_rate_e'].append(
            levenshtein(ra_ref, ra_e)
        )
        analysis['lev_course_rate_e_n'].append(
            levenshtein(ra_ref, ra_e) / len(ra_ref)
        )
        analysis['kl_course_rate_e'].append(
            kl_divergence(ra_ref, ra_e)
        )
        
    return analysis
