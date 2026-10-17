"""Counter-based random numbers shared by every sampler.

Each uniform is a SplitMix64 hash of (stream key, seed, trajectory j, step t, slot). Two methods
run with the same stream and seeds therefore see the same uniforms at the same (seed, j, t), so
they are compared under common random numbers. Seeds also run in parallel without sharing state.
"""
import hashlib

import numpy as np
from numba import njit

ROOT = 20261012                     # root of every stream key
SLOT_START, SLOT_ACTION, SLOT_NEXT_STATE, SLOT_NOISE_1, SLOT_NOISE_2 = 0, 1, 2, 3, 4

_C1 = np.uint64(0xBF58476D1CE4E5B9)
_C2 = np.uint64(0x94D049BB133111EB)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def stream_key(stream, root=ROOT):
    """64-bit key of a named random stream."""
    digest = hashlib.blake2b(f"{root}:{stream}".encode(), digest_size=8).digest()
    return np.uint64(int.from_bytes(digest, "little"))


@njit(cache=True)
def uniform(key, seed, j, t, slot):
    """Uniform on [0, 1) at position (seed < 2^16, trajectory j < 2^20, step t < 2^8, slot < 2^4)."""
    idx = ((np.uint64(seed) * np.uint64(1048576) + np.uint64(j)) * np.uint64(256)
           + np.uint64(t)) * np.uint64(16) + np.uint64(slot)
    z = key + (idx + np.uint64(1)) * _GOLDEN
    z = (z ^ (z >> np.uint64(30))) * _C1
    z = (z ^ (z >> np.uint64(27))) * _C2
    z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True)
def sample_from_cdf(cdf, u):
    """Smallest i with u < cdf[i]; if rounding leaves u above cdf[-1], the last index with mass."""
    n = cdf.shape[0]
    for i in range(n):
        if u < cdf[i]:
            return i
    for i in range(n - 1, 0, -1):
        if cdf[i] > cdf[i - 1]:
            return i
    return 0


@njit(cache=True)
def sample_from_log_probs(logp, u):
    """Index drawn with probabilities exp(logp), using the uniform u."""
    acc = 0.0
    last = 0
    for a in range(logp.shape[0]):
        p = np.exp(logp[a])
        if p > 0.0:
            last = a
        acc += p
        if u < acc:
            return a
    return last


@njit(cache=True)
def env_step(P_cum, R, reward_std, s, a, key, seed, j, t):
    """Next state and reward for action a in state s (Gaussian reward noise via Box-Muller)."""
    s2 = sample_from_cdf(P_cum[s, a], uniform(key, seed, j, t, SLOT_NEXT_STATE))
    r = R[s, a, s2]
    sd = reward_std[s, a, s2]
    if sd > 0.0:
        u1 = uniform(key, seed, j, t, SLOT_NOISE_1)
        u2 = uniform(key, seed, j, t, SLOT_NOISE_2)
        r += sd * np.sqrt(-2.0 * np.log(1.0 - u1)) * np.cos(2.0 * np.pi * u2)
    return s2, r
