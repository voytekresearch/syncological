import numpy as np


def thal_spikes(t, f, N, k, prng=None):
    """Create synchronous bursts (1 ms variance) of thalamic-ish spike

    Params
    ------
    t : numeric
        Simulation time (s)
    f : numeric
        Oscillation frequency (Hz)
    N : numeric
        Number of neurons
    k : numeric
        Number of neuron to spike at a time
    """

    if k > N:
        raise ValueError("k is larger than N")
    if f < 0:
        raise ValueError("f must be greater then 0")
    if N < 0:
        raise ValueError("N must be greater then 0")
    if k < 0:
        raise ValueError("k must be greater then 0")

    if prng is None:
        prng = np.random.RandomState()

    spikes_stdev = 1 / 1000.0 # ms

    # Locate about where the pulses of spikes will go, at f,
    dt = 1 / float(f)
    n_pulses = int(t * f)
    pulses = []
    t_p = 0
    for _ in range(n_pulses):
        t_p += dt

        # Gaurd against negative times
        if t_p > (3 * spikes_stdev):
            pulses.append(t_p)

    # and fill in the pulses with Gaussin distributed spikes.
    Ns = range(N)
    times = []
    idx = []
    for t in pulses:
        times += list(t + prng.normal(0, spikes_stdev, k))

        # Assign spikes to random neurons, at most
        # one spike / neuron
        prng.shuffle(Ns)
        idx += list(Ns)[0:k]

    times = np.array(times)
    idx = np.array(idx)

    # Just in case any negative time any slipped trough
    times[times < 0] = 0.0

    return times, idx


def gaussian_impulse(t, min_t, max_t, stdev, N, k, decimals=5, prng=None):
    """Create a bursts of spikes around t.

    Params
    ------
    t : numeric (s)
        Simulation time where the center of the bursrt should lie
    min_t : numeric (s)
        Minumum allowed t
    max_t : numeric (s)
        Maximum allowed t
    stdev : numeric (s)
        Spike ISI stat dev
    N : numeric
        Number of neurons
    k : numeric
        Number of spikes (approximately)
    decimals : int
        Resolution of spike times
    prng : None, RandomState
        Controls random seed. If None a new RandomState()
        is generated with each call.
    """

    if N < 0:
        raise ValueError("N must be greater then 0")
    if k < 0:
        raise ValueError("k must be greater then 0")
    if stdev < 0:
        raise ValueError("stdev must be greater then 0")

    if prng is None:
        prng = np.random.RandomState()

    decimals = int(decimals)

    # Create the burst and set the time resolution
    times = prng.normal(t, stdev, k)
    times = np.round(times, decimals=decimals)

    # Remove negative time
    # and times that are too
    # small or big
    mask = times > 0
    times = times[mask]
    mask = times >= min_t
    times = times[mask]
    mask = times <= max_t
    times = times[mask]

    # Randomly assign neurons to times
    idx = prng.random_integers(0, N, size=len(times))

    return times, idx

