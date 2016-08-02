from __future__ import division

import numpy as np
import os, sys
import pyspike as spk

from collections import defaultdict
from pyspike import SpikeTrain, isi_distance, spike_sync
from fakespikes import neurons, util, rates
from fakespikes.util import (kappa, ts_sort, levenshtein, kl_divergence,
                             rate_code, spike_window_code, coincidence_code,
                             bin_times)
from collections import defaultdict, Counter
from random import shuffle

from syncological.util import progressbar
from joblib import Memory
from tempfile import mkdtemp
from misshapen.shape import findpt
from pacpy.filt import firf
from scipy.signal import hilbert

from collections import defaultdict
from random import shuffle

# -- Setup memoization
CACHEDIR = mkdtemp()
memory = Memory(cachedir=CACHEDIR, verbose=0)


def load_results(number, path, to_load='all'):
    if to_load == 'all':
        to_load = [
            "exampletrace_i", "i_ee", "i_i", "i_ii", "j_ee", "j_i", "j_ii",
            "poprates_i", "spiketimes_e", "spiketimes_back_e",
            "spiketimes_stim", "sta_i", "exampletrace_e", "i_e", "i_ei",
            "i_ie", "j_e", "j_ei", "j_ie", "lfp", "poprates_e", "psd",
            "spiketimes_i", "spiketimes_back_i", "sta_e"
        ]

    result = {}
    for l in to_load:
        f = os.path.join(path, str(number) + "_" + l + ".csv")
        result[l] = np.loadtxt(f, delimiter=",")

    return result


def load_trial(number, trial, path, to_load='all'):
    if to_load == 'all':
        to_load = [
            "exampletrace_i", "i_ee", "i_i", "i_ii", "j_ee", "j_i", "j_ii",
            "poprates_i", "spiketimes_e", "spiketimes_back_e",
            "spiketimes_stim", "sta_i", "exampletrace_e", "i_e", "i_ei",
            "i_ie", "j_e", "j_ei", "j_ie", "lfp", "poprates_e", "psd",
            "spiketimes_i", "spiketimes_back_i", "sta_e"
        ]

    result = {}
    for l in to_load:
        f = os.path.join(path, "{}_k{}_{}.csv".format(number, trial, l))
        result[l] = np.loadtxt(f, delimiter=",")

    return result


@memory.cache
def load_exp(code, n_trials, path, to_load, offset=0.1):
    tlast_e = 0.0
    tlast_stim = 0.0

    times = load_trial(code, 0, path,
                       to_load=['exampletrace_e'])['exampletrace_e'][:, 0]

    trials = defaultdict(dict)
    for k in range(n_trials):
        res = load_trial(code, k, path, to_load=to_load)
        trials[k]['ns_e'] = res['spiketimes_e'][:, 0].astype(int)
        trials[k]['ts_e'] = res['spiketimes_e'][:, 1]
        trials[k]['ns_stim'] = res['spiketimes_stim'][:, 0].astype(int)
        trials[k]['ts_stim'] = res['spiketimes_stim'][:, 1]

        trials[k]['lfp'] = res['lfp']

        # renorm time
        tmax = trials[k]['ts_e'].max()
        trials[k]['ts_e'] -= tlast_e
        tlast_e = tmax

        tmax = trials[k]['ts_stim'].max()
        trials[k]['ts_stim'] -= tlast_stim
        tlast_stim = tmax

        # drop before offset
        trials[k]['lfp'] = trials[k]['lfp'][times > offset]

        m = trials[k]['ts_e'] > offset
        trials[k]['ts_e'] = trials[k]['ts_e'][m]
        trials[k]['ns_e'] = trials[k]['ns_e'][m]

    return trials


def sample_neurons(trials, n_neurons):
    nss = [set(trial['ns_e']) for trial in trials.values()]
    common = nss[0]
    for ns in nss[1:]:
        common.intersection_update(ns)
    common = list(common)
    shuffle(common)

    return common[:n_neurons]


def top_connections(trials, c, conn='e'):
    """Keep only data with the top c synaptic inputs"""

    idx = trials[0]['j_' + conn]
    cnt = Counter()
    for x in idx:
        cnt[int(x)] += 1

    top_c = cnt.most_common(n)

    for i in len(trials):
        ns, ts = trials[i]['ns_' + conn], trials[i]['ts_' + conn]

        m = np.zeros_like(ns, dtype=np.bool)
        for n, _ in top_c:
            m = np.logical_or(m, n == ns)

        trials[i]['ns_' + conn], trials[i]['ts_' + conn] = ns[m], ts[m]

    return trials


def trial_coding(trials, ref='e'):
    """Compare coding properties across trials."""

    # Define references
    ns_ref, ts_ref = trials[0]['ns_' + ref], trials[0]['ts_' + ref]
    a_spike = util.spike_time_code(ts_ref, scale=1000, decimals=4)
    a_rate = util.spike_time_code(ts_ref, scale=1000, decimals=1)
    a_order, _ = util.ts_sort(ns_ref, ts_ref)

    # -
    precs = []
    lev_spike = []
    lev_rate = []
    lev_order = []
    for k, res in enumerate(trials.values()[1:]):
        ns, ts = res['ns_e'], res['ts_e']
        _, prec = util.precision(ns, ts, ns_ref, ts_ref, combine=True)
        precs.append(prec)

        b_spike = util.spike_time_code(ts, scale=1000)
        b_rate = util.spike_time_code(ts, scale=1000, decimals=1)
        b_order, _ = util.ts_sort(ns, ts)

        lev_spike.append(util.levenshtein(a_spike, b_spike) /
                         float(a_spike.size))
        lev_rate.append(util.levenshtein(a_rate, b_rate) / float(a_rate.size))
        lev_order.append(util.levenshtein(a_order, b_order) /
                         float(a_rate.size))

    precs = np.asarray(precs)
    lev_rate = np.asarray(lev_rate)
    lev_spike = np.asarray(lev_spike)
    lev_order = np.asarray(lev_order)

    return precs, lev_spike, lev_order, lev_rate


def trial_S(trials, ref='e', t_min=None, t_max=None):
    ns_ref, ts_ref = trials[0]['ns_' + ref], trials[0]['ts_' + ref]

    if t_min is None:
        t_min = ts_ref.min()
    if t_max is None:
        t_max = ts_ref.max()

    t_range = (t_min, t_max)

    isi = []
    sync = []
    for k, res in enumerate(trials.values()[1:]):
        ns, ts = res['ns_e'], res['ts_e']

        sto = spk.SpikeTrain(ts, t_range)
        sto_ref = spk.SpikeTrain(ts_ref, t_range)

        sto.sort()
        sto_ref.sort()

        isi.append(spk.isi_distance(sto_ref, sto))
        sync.append(spk.spike_sync(sto_ref, sto))

    return np.asarray(isi), np.asarray(sync)


def trial_kl(trials, ref='e'):
    """Compare coding properties across trials."""

    # Define references
    ns_ref, ts_ref = trials[0]['ns_' + ref], trials[0]['ts_' + ref]
    a_spike = util.spike_time_code(ts_ref, scale=1000, decimals=4)
    a_rate = util.spike_time_code(ts_ref, scale=1000, decimals=1)
    a_order, _ = util.ts_sort(ns_ref, ts_ref)

    # Run
    kl_spike = []
    kl_rate = []
    kl_order = []
    for k, res in enumerate(trials.values()[1:]):
        ns, ts = res['ns_e'], res['ts_e']

        b_spike = util.spike_time_code(ts, scale=1000)
        b_rate = util.spike_time_code(ts, scale=1000, decimals=1)
        b_order, _ = util.ts_sort(ns, ts)

        kl_spike.append(util.kl_divergence(a_spike, b_spike))
        kl_rate.append(util.kl_divergence(a_rate, b_rate))
        kl_order.append(util.kl_divergence(a_order, b_order))

    kl_rate = np.asarray(kl_rate)
    kl_spike = np.asarray(kl_spike)
    kl_order = np.asarray(kl_order)

    return kl_spike, kl_order, kl_rate


def power_coding(trials, times, lfps, amps, peaks, ref='e', relative=False):
    """Compare coding properties as gamma power fluctuates."""

    n_trials = len(trials)

    precs = []
    lev_s = []
    lev_r = []
    lev_o = []
    pows = []
    skip_i = 0
    skip_r = 0
    trials_k = []

    ns_ref, ts_ref = trials[0]['ns_' + ref], trials[0]['ts_' + ref]
    for k in range(1, n_trials):
        res = trials[k]
        peak = peaks[k]
        amp = amps[k]
        ns_1, ts_1 = res['ns_e'], res['ts_e']

        for i in range(len(peak) - 1):
            t0 = times[peak[i]]
            t1 = times[peak[i + 1]]

            # Build filter and select data
            m = np.logical_and(ts_1 >= t0, ts_1 < t1)
            ts_i = ts_1[m]
            ns_i = ns_1[m]

            if ts_i.size == 0:
                skip_i += 1
                continue

            m = np.logical_and(ts_ref >= t0, ts_ref < t1)
            ts_r = ts_ref[m]
            ns_r = ns_ref[m]

            if ts_r.size == 0:
                skip_r += 1
                continue

            # Reset to relative timing
            if relative:
                ts_i = ts_i - ts_i.min()
                ts_r = ts_r - ts_r.min()

            # Precision and fidelity.
            _, prec = util.precision(ns_i, ts_i, ns_r, ts_r, combine=True)
            precs.append(prec)

            a_spike = util.spike_time_code(ts_r, scale=1000, decimals=4)
            b_spike = util.spike_time_code(ts_i, scale=1000, decimals=4)
            lev_s.append(util.levenshtein(a_spike, b_spike) /
                         float(a_spike.size))

            a_rate = util.spike_time_code(ts_r, scale=1000, decimals=1)
            b_rate = util.spike_time_code(ts_i, scale=1000, decimals=1)
            lev_r.append(util.levenshtein(a_rate, b_rate) / float(a_rate.size))

            a_order, _ = util.ts_sort(ns_r, ts_r)
            b_order, _ = util.ts_sort(ns_i, ts_i)
            lev_o.append(util.levenshtein(a_order, b_order) /
                         float(a_order.size))

            # power
            assert times.shape == amp.shape, "lfp and time mismatch"
            m = np.logical_and(times >= t0, times < t1)
            pows.append(np.mean(amp[m]))

            trials_k.append(k)

    precs = np.asarray(precs)
    lev_s = np.asarray(lev_s)
    lev_o = np.asarray(lev_o)
    lev_r = np.asarray(lev_r)
    pows = np.asarray(pows)
    trials_k = np.asarray(trials_k)

    return precs, lev_s, lev_o, lev_r, pows, trials_k, skip_i, skip_r


def phase_coding(trials, times, peaks, troughs, ref='e', relative=False):
    """Compare coding properties across gamma phase"""

    n_trials = len(trials)

    precs_rise = []
    precs_fall = []
    lev_s_rise = []
    lev_s_fall = []
    lev_o_rise = []
    lev_o_fall = []
    lev_r_rise = []
    lev_r_fall = []

    skip_r = 0
    skip_f = 0

    trials_k = []

    ns_ref, ts_ref = trials[0]['ns_' + ref], trials[0]['ts_' + ref]
    for k in range(1, n_trials):
        res = trials[k]
        peak = peaks[k]
        trough = troughs[k]

        ns_1, ts_1 = res['ns_e'], res['ts_e']

        for i in range(len(peak) - 1):
            t0 = times[peak[i]]
            tm = times[trough[i]]
            t1 = times[peak[i + 1]]

            # Build filter and select data
            m_rise = np.logical_and(ts_1 >= t0, ts_1 < tm)
            m_fall = np.logical_and(ts_1 >= tm, ts_1 < t1)

            ts_i_rise = ts_1[m_rise]
            ns_i_rise = ts_1[m_rise]
            ts_i_fall = ts_1[m_fall]
            ns_i_fall = ts_1[m_fall]

            if ts_i_rise.size == 0:
                skip_r += 1
                continue
            if ts_i_fall.size == 0:
                skip_f += 1
                continue

            m_rise = np.logical_and(ts_ref >= t0, ts_ref < tm)
            m_fall = np.logical_and(ts_ref >= tm, ts_ref < t1)

            ts_r_rise = ts_ref[m_rise]
            ns_r_rise = ts_ref[m_rise]
            ts_r_fall = ts_ref[m_fall]
            ns_r_fall = ts_ref[m_fall]

            if ts_r_rise.size == 0:
                skip_r += 1
                continue
            if ts_r_fall.size == 0:
                skip_f += 1
                continue

            # Reset to relative timing
            if relative:
                ts_i_rise = ts_i_rise - ts_i_rise.min()
                ts_r_rise = ts_r_rise - ts_r_rise.min()
                ts_i_fall = ts_i_fall - ts_i_fall.min()
                ts_r_fall = ts_r_fall - ts_r_fall.min()

            # Precision and fidelity.
            # Rise
            _, prec = util.precision(ns_i_rise,
                                     ts_i_rise,
                                     ns_r_rise,
                                     ts_r_rise,
                                     combine=True)
            precs_rise.append(prec)

            a_spike = util.spike_time_code(ts_r_rise, scale=1000, decimals=4)
            b_spike = util.spike_time_code(ts_i_rise, scale=1000, decimals=4)
            lev_s_rise.append(util.levenshtein(a_spike, b_spike) /
                              float(a_spike.size))

            a_rate = util.spike_time_code(ts_r_rise, scale=1000, decimals=1)
            b_rate = util.spike_time_code(ts_i_rise, scale=1000, decimals=1)
            lev_r_rise.append(util.levenshtein(a_rate, b_rate) /
                              float(a_rate.size))

            a_order, _ = util.ts_sort(ns_r_rise, ts_r_rise)
            b_order, _ = util.ts_sort(ns_i_rise, ts_i_rise)
            lev_o_rise.append(util.levenshtein(a_order, b_order) /
                              float(a_rate.size))

            # Fall
            _, prec = util.precision(ns_i_fall,
                                     ts_i_fall,
                                     ns_r_fall,
                                     ts_r_fall,
                                     combine=True)
            precs_fall.append(prec)

            a_spike = util.spike_time_code(ts_r_fall, scale=1000, decimals=4)
            b_spike = util.spike_time_code(ts_i_fall, scale=1000, decimals=4)
            lev_s_fall.append(util.levenshtein(a_spike, b_spike) /
                              float(a_spike.size))

            a_rate = util.spike_time_code(ts_r_fall, scale=1000, decimals=4)
            b_rate = util.spike_time_code(ts_i_fall, scale=1000, decimals=4)
            lev_r_fall.append(util.levenshtein(a_rate, b_rate) /
                              float(a_rate.size))

            a_order, _ = util.ts_sort(ns_r_fall, ts_r_fall)
            b_order, _ = util.ts_sort(ns_i_fall, ts_i_fall)
            lev_o_fall.append(util.levenshtein(a_order, b_order) /
                              float(a_rate.size))

            trials_k.append(k)

    precs_rise = np.asarray(precs_rise)
    precs_fall = np.asarray(precs_fall)
    lev_s_rise = np.asarray(lev_s_rise)
    lev_s_fall = np.asarray(lev_s_fall)
    lev_o_rise = np.asarray(lev_o_rise)
    lev_o_fall = np.asarray(lev_o_fall)
    lev_r_rise = np.asarray(lev_r_rise)
    lev_r_fall = np.asarray(lev_r_fall)

    return (precs_rise, precs_fall, lev_s_rise, lev_s_fall, lev_o_rise,
            lev_o_fall, lev_r_rise, lev_r_fall, trials_k, skip_f, skip_r)


def extract_lfps(trials):
    """Extract LFP data from trials"""

    n_trials = len(trials)

    # Get lfp
    lfps = []
    for k in range(n_trials):
        lfps.append(trials[k]['lfp'])

    return lfps


def gamma_amplitude(lfps, fs):
    """Extract gamma amplitude from LFPs"""

    amps = []
    for i, lfp in enumerate(lfps):
        x = firf(lfp, (20, 100), fs=fs, w=3, rmvedge=False)
        amps.append(np.abs(hilbert(x)))

    return amps


def peak_troughs(lfps, fs):
    """Find peaks and troughs in the LFPs"""

    peaks, troughs = [], []
    for i, lfp in enumerate(lfps):
        p, tr = findpt(lfp, (20, 100), Fs=fs, boundary=100)
        peaks.append(p)
        troughs.append(tr)

    return peaks, troughs


def stim_seperation(trials_1, trials_2, dt, T=2.1):
    """Estimate how different two stimuli are"""

    if len(trials_1) != len(trials_2):
        raise ValueError("trials must be the same length.")

    n_trials = len(trials_1)

    seps = []
    fracs = []
    for k in range(n_trials):
        ns_1, ts_1 = trials_1[k]['ns_e'], trials_1[k]['ts_e']
        ns_2, ts_2 = trials_2[k]['ns_e'], trials_2[k]['ts_e']

        sep, _, _ = util.seperation(ns_1, ts_1, ns_2, ts_2, dt, T=T)
        seps.append(sep)

        # fraction of time > 1.0
        frac = np.sum(sep > 1.0) / float(sep.size)
        fracs.append(frac)

    return seps, fracs


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
    skipped = []
    analysis = defaultdict(list)
    for m in progressbar(models):
        if ref == m:
            continue

        try:
            model_data = load_results(m, path)
        except:
            skipped.append(m)
            continue

        analysis['models'].append(m)
        ns_e = model_data['spiketimes_e'][:, 0]
        ts_e = model_data['spiketimes_e'][:, 1]

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

        analysis['lev_spikeorder'].append(levenshtein(
            list(ordered_ref), list(ordered_e)))
        analysis['lev_spikeorder_n'].append(levenshtein(
            list(ordered_ref), list(ordered_e)) / len(ordered_ref))
        analysis['kl_spikeorder'].append(kl_divergence(
            list(ordered_ref), list(ordered_e)))

        # - Spike timing code, as packets in a 1 ms window
        pack_e, _, _ = spike_window_code(ts_e, ns_e, dt=1e-3)
        pack_ref, _, _ = spike_window_code(ts_ref, ns_ref, dt=1e-3)
        analysis['lev_spikepack'].append(levenshtein(
            list(pack_ref), list(pack_e)))
        analysis['lev_spikepack_n'].append(levenshtein(
            list(pack_ref), list(pack_e)) / len(pack_ref))
        analysis['kl_spikepack'].append(kl_divergence(
            list(pack_ref), list(pack_e)))

        # - CC count code
        cc_ref, _ = coincidence_code(ts_ref, ns_ref, 1e-3)
        cc_e, _ = coincidence_code(ts_e, ns_e, 1e-3)

        analysis['lev_cc'].append(levenshtein(list(cc_ref), list(cc_e)))
        analysis['lev_cc_n'].append(levenshtein(
            list(cc_ref), list(cc_e)) / len(cc_ref))
        analysis['kl_cc'].append(kl_divergence(list(cc_ref), list(cc_e)))

        # - Rate order not magnitude
        # 'fine rate' order, AMPA time scale
        ra_e, _, _ = rate_code(ts_e, (time0, time), 5e-3)
        ra_ref, _, _ = rate_code(ts_ref, (time0, time), 5e-3)

        analysis['lev_fine_rateorder'].append(levenshtein(ra_ref, ra_e))
        analysis['lev_fine_rateorder_n'].append(levenshtein(ra_ref, ra_e) /
                                                len(ra_ref))
        analysis['kl_fine_rateorder'].append(kl_divergence(ra_ref, ra_e))

        # 'course rate' order, NMDA time scale
        ra_e, _, _ = rate_code(ts_e, (time0, time), 45e-3)
        ra_ref, _, _ = rate_code(ts_ref, (time0, time), 45e-3)

        analysis['lev_course_rateorder'].append(levenshtein(ra_ref, ra_e))
        analysis['lev_course_rateorder_n'].append(levenshtein(ra_ref, ra_e) /
                                                  len(ra_ref))
        analysis['kl_course_rateorder'].append(kl_divergence(
            list(ra_ref), list(ra_e)))

        # - Rate magnitude code, both coarse and fine
        # AMPA
        t_range = (time0, time)

        _, binned_ref = bin_times(ts_ref, t_range, 5e-3)
        _, binned_e = bin_times(ts_e, t_range, 5e-3)
        binned_e = binned_e.astype(int)
        binned_ref = binned_ref.astype(int)

        analysis['lev_fine_rate'].append(levenshtein(binned_ref, binned_e))
        analysis['lev_fine_rate_n'].append(levenshtein(binned_ref, binned_e) /
                                           len(binned_ref))
        analysis['kl_fine_rate'].append(kl_divergence(
            list(binned_ref), list(binned_e)))

        # NMDA
        _, binned_ref = bin_times(ts_ref, t_range, 45e-3)
        _, binned_e = bin_times(ts_e, t_range, 45e-3)
        binned_e = binned_e.astype(int)
        binned_ref = binned_ref.astype(int)

        analysis['lev_coarse_rate'].append(levenshtein(binned_ref, binned_e))
        analysis['lev_coarse_rate_n'].append(
            levenshtein(binned_ref, binned_e) / len(binned_ref))
        analysis['kl_coarse_rate'].append(kl_divergence(
            list(binned_ref), list(binned_e)))

        # - Norm rate magnitude code, both coarse and fine
        # AMPA
        _, binned_ref = bin_times(ts_ref, t_range, 5e-3)
        _, binned_e = bin_times(ts_e, t_range, 5e-3)

        binned_ref = binned_ref / float(binned_ref.max())
        binned_e = binned_e / float(binned_e.max())
        binned_ref = np.digitize(binned_ref, np.linspace(0, 1, 100))
        binned_e = np.digitize(binned_e, np.linspace(0, 1, 100))

        analysis['lev_fine_normrate'].append(levenshtein(binned_ref, binned_e))
        analysis['lev_fine_normrate_n'].append(
            levenshtein(binned_ref, binned_e) / len(binned_ref))
        analysis['kl_fine_normrate'].append(kl_divergence(
            list(binned_ref), list(binned_e)))

        # NMDA
        _, binned_ref = bin_times(ts_ref, t_range, 45e-3)
        _, binned_e = bin_times(ts_e, t_range, 45e-3)

        binned_ref = binned_ref / float(binned_ref.max())
        binned_e = binned_e / float(binned_e.max())
        binned_ref = np.digitize(binned_ref, np.linspace(0, 1, 100))
        binned_e = np.digitize(binned_e, np.linspace(0, 1, 100))
        analysis['lev_coarse_normrate'].append(levenshtein(binned_ref,
                                                           binned_e))
        analysis['lev_coarse_normrate_n'].append(
            levenshtein(binned_ref, binned_e) / len(binned_ref))
        analysis['kl_coarse_normrate'].append(kl_divergence(
            list(binned_ref), list(binned_e)))

    return analysis, skipped
