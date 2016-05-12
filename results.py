from __future__ import division

import numpy as np
import os, sys
from collections import defaultdict
from pyspike import SpikeTrain, isi_distance, spike_sync
from fakespikes.util import (kappa, ts_sort, levenshtein, 
                             kl_divergence, rate_code, 
                             spike_window_code, coincidence_code, 
                             bin_times)
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


def compare(ref, models, path, drop_before=0.1, ref_path=None, n_syn=0):
    """Compate models to a reference model"""

    if n_syn != 0:
        raise NotImplementedError("TODO")

    # --
    # load ref
    if ref_path is None:
        ref_path = path
    ref_data = load_results(ref, ref_path)

    # unpack then
    ns_ref = ref_data['spiketimes_e'][:, 0]
    ts_ref = ref_data['spiketimes_e'][:, 1]
    
    # filter - drop before drop_before
    mask = ts_ref >= drop_before
    ns_ref, ts_ref = ns_ref[mask], ts_ref[mask]

    time0 = ts_ref.min()
    time = ts_ref.max()

    # --
    analysis = defaultdict(list)
    for m in progressbar(models):
        if ref == m:
            continue
        analysis['models'].append(m)

        # unpack then
        model_data = load_results(m, path)
        ns_e = model_data['spiketimes_e'][:,0]
        ts_e = model_data['spiketimes_e'][:,1]
        
        # filter - drop before drop_before
        mask = ts_e >= drop_before
        ns_e, ts_e = ns_e[mask], ts_e[mask]

        # - sync
        sto_e = SpikeTrain(ts_e, (time0, time))
        sto_ref = SpikeTrain(ts_ref, (time0, time))
        sto_e.sort()
        sto_ref.sort()

        analysis['s_sync'].append(spike_sync(sto_ref, sto_e))
        
        # --
        # lev and KL distance
        # - spike order code
        ordered_e, _ = ts_sort(ns_e, ts_e)
        ordered_ref, _ = ts_sort(ns_ref, ts_ref)

        analysis['lev_spikeorder'].append(
            levenshtein(
                list(ordered_ref), 
                list(ordered_e))
        )
        analysis['lev_spikeorder_n'].append(
            levenshtein(
                list(ordered_ref), 
                list(ordered_e)) / len(ordered_ref)
        )
        analysis['kl_spikeorder'].append(
            kl_divergence(list(ordered_ref), list(ordered_e))
        )
        
        # - Spike timing code, as packets in a 1 ms window
        pack_e, _, _ = spike_window_code(ts_e, ns_e, dt=1e-3)
        pack_ref, _, _ = spike_window_code(ts_ref, ns_ref, dt=1e-3)
        analysis['lev_spikepack'].append(
            levenshtein(
                list(pack_ref), 
                list(pack_e))
        )
        analysis['lev_spikepack_n'].append(
            levenshtein(
                list(pack_ref), 
                list(pack_e)) / len(pack_ref)
        )
        analysis['kl_spikepack'].append(
            kl_divergence(list(pack_ref), list(pack_e))
        )

        # - CC count code
        cc_ref, _ = coincidence_code(ts_ref, ns_ref, 1e-3)
        cc_e, _ = coincidence_code(ts_e, ns_e, 1e-3)

        analysis['lev_cc'].append(
            levenshtein(
                list(cc_ref), 
                list(cc_e))
        )
        analysis['lev_cc_n'].append(
            levenshtein(
                list(cc_ref), 
                list(cc_e)) / len(cc_ref)
        )
        analysis['kl_cc'].append(
            kl_divergence(list(cc_ref), list(cc_e))
        )
        
        # - Rate order not magnitude
        # 'fine rate' order, AMPA time scale
        ra_e, _, _ = rate_code(ts_e, (time0, time), 5e-3)
        ra_ref, _, _ = rate_code(ts_ref, (time0, time), 5e-3)

        analysis['lev_fine_rateorder'].append(
            levenshtein(ra_ref, ra_e)
        )
        analysis['lev_fine_rateorder_n'].append(
            levenshtein(ra_ref, ra_e) / len(ra_ref)
        )
        analysis['kl_fine_rateorder'].append(
            kl_divergence(ra_ref, ra_e)
        )
        
        # 'course rate' order, NMDA time scale
        ra_e, _, _ = rate_code(ts_e, (time0, time), 45e-3)
        ra_ref, _, _ = rate_code(ts_ref, (time0, time), 45e-3)

        analysis['lev_course_rateorder'].append(
            levenshtein(ra_ref, ra_e)
        )
        analysis['lev_course_rateorder_n'].append(
            levenshtein(ra_ref, ra_e) / len(ra_ref)
        )
        analysis['kl_course_rateorder'].append(
            kl_divergence(list(ra_ref), list(ra_e))
        )
        
        # - Rate magnitude code, both coarse and fine
        # AMPA
        t_range = (time0, time)

        _, binned_ref = bin_times(ts_ref, t_range, 5e-3)
        _, binned_e = bin_times(ts_e, t_range, 5e-3)
        binned_e = binned_e.astype(int)
        binned_ref = binned_ref.astype(int)

        analysis['lev_fine_rate'].append(
            levenshtein(binned_ref, binned_e)
        )
        analysis['lev_fine_rate_n'].append(
            levenshtein(binned_ref, binned_e) / len(binned_ref)
        )
        analysis['kl_fine_rate'].append(
            kl_divergence(list(binned_ref), list(binned_e))
        )
        
        # NMDA
        _, binned_ref = bin_times(ts_ref, t_range, 45e-3)
        _, binned_e = bin_times(ts_e, t_range, 45e-3)
        binned_e = binned_e.astype(int)
        binned_ref = binned_ref.astype(int)

        analysis['lev_coarse_rate'].append(
            levenshtein(binned_ref, binned_e)
        )
        analysis['lev_coarse_rate_n'].append(
            levenshtein(binned_ref, binned_e) / len(binned_ref)
        )
        analysis['kl_coarse_rate'].append(
            kl_divergence(list(binned_ref), list(binned_e))
        )
        
        # - Norm rate magnitude code, both coarse and fine
        # AMPA
        _, binned_ref = bin_times(ts_ref, t_range, 5e-3)
        _, binned_e = bin_times(ts_e, t_range, 5e-3)

        binned_ref = binned_ref / float(binned_ref.max())
        binned_e = binned_e / float(binned_e.max())
        binned_ref = np.digitize(binned_ref, np.linspace(0, 1, 100))
        binned_e = np.digitize(binned_e, np.linspace(0, 1, 100))
        
        analysis['lev_fine_normrate'].append(
            levenshtein(binned_ref, binned_e)
        )
        analysis['lev_fine_normrate_n'].append(
            levenshtein(binned_ref, binned_e) / len(binned_ref)
        )
        analysis['kl_fine_normrate'].append(
            kl_divergence(list(binned_ref), list(binned_e))
        )
        
        # NMDA
        _, binned_ref = bin_times(ts_ref, t_range, 45e-3)
        _, binned_e = bin_times(ts_e, t_range, 45e-3)

        binned_ref = binned_ref / float(binned_ref.max())
        binned_e = binned_e / float(binned_e.max())
        binned_ref = np.digitize(binned_ref, np.linspace(0, 1, 100))
        binned_e = np.digitize(binned_e, np.linspace(0, 1, 100))
        analysis['lev_coarse_normrate'].append(
            levenshtein(binned_ref, binned_e)
        )
        analysis['lev_coarse_normrate_n'].append(
            levenshtein(binned_ref, binned_e) / len(binned_ref)
        )
        analysis['kl_coarse_normrate'].append(
            kl_divergence(list(binned_ref), list(binned_e))
        )

    return analysis
