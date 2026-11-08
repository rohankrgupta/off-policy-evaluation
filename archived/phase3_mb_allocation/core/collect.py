"""Collect data from a behavior target under a fixed budget of environment steps, and estimate v(pi_e).

The target is a softmax policy: one per seed or shared, stationary (S, A) or time-indexed (L, S, A).
Actions are drawn i.i.d. from it (alpha = 0) or with ROS tracking it (alpha > 0, stationary targets
only, per-state counts as in [2]).

At each step checkpoint T (the first episode end at or after T steps) the kernel records:
  * MC: the mean return of this run's episodes (unbiased only when the target is pi_e);
  * MB: the model-based estimate. It fits P(s'|s,a) and the mean reward from counts pooled over t,
    then evaluates pi_e exactly by DP in the fitted model. Pairs never visited count as value 0.

Earlier data (`init`, e.g. stage-1 counts) are pooled into MB's model, so they are not wasted.
"""
import numpy as np
from numba import njit, prange

from phase1_reproduction.core import exact
from phase1_reproduction.core.rng import (SLOT_ACTION, SLOT_START, env_step, sample_from_cdf,
                                          sample_from_log_probs, stream_key, uniform)
from phase1_reproduction.core.sampling import _log_softmax_into

STAT_KEYS = ("counts", "reward_sums", "reward_sq_sums", "starts")


@njit(cache=True)
def _mb_estimate(N, RS, starts, pi_e, terminal, horizon, gamma, V):
    """Value of pi_e in the model fitted to transition counts N and reward sums RS (both (S, A, S))."""
    S, A = pi_e.shape
    V[horizon] = 0.0
    for t in range(horizon - 1, -1, -1):
        for s in range(S):
            v = 0.0
            if not terminal[s]:
                for a in range(A):
                    n_sa = 0.0
                    total = 0.0
                    for s2 in range(S):
                        n_sa += N[s, a, s2]
                        total += RS[s, a, s2] + gamma * N[s, a, s2] * V[t + 1, s2]
                    if n_sa > 0.0:
                        v += pi_e[s, a] * total / n_sa
            V[t, s] = v
    return (starts * V[0]).sum() / starts.sum()


@njit(parallel=True, cache=True)
def _kernel(P_cum, R, reward_std, terminal, d0_cum, horizon, gamma, pi_e, theta, prob, alpha,
            step_checkpoints, seeds, key, est, init, init_starts, has_init, stats, stat_starts):
    S, A = pi_e.shape
    for i in prange(seeds.shape[0]):
        seed = seeds[i]
        ti = np.int64(0)                    # this seed's target: shared or per seed
        if theta.shape[0] > 1:
            ti = np.int64(i)
        timed = theta.shape[1] > 1
        C = np.zeros((S, A))                # ROS counts: this run's data only
        m = np.zeros(S)
        k = 0.0
        N = np.zeros((S, A, S))             # model statistics, pooled with earlier data
        RS = np.zeros((S, A, S))
        RQ = np.zeros((S, A, S))
        starts = np.zeros(S)
        if has_init:
            N[:] = init[0, i]
            RS[:] = init[1, i]
            RQ[:] = init[2, i]
            starts[:] = init_starts[i]
        logits = np.empty(A)
        logp = np.empty(A)
        V = np.zeros((horizon + 1, S))
        steps = 0
        episodes = 0
        sum_returns = 0.0
        ck = 0
        j = 0
        while ck < step_checkpoints.shape[0]:
            s = sample_from_cdf(d0_cum, uniform(key, seed, j, 0, SLOT_START))
            starts[s] += 1.0
            g = 0.0
            disc = 1.0
            for t in range(horizon):
                if terminal[s]:
                    break
                tt = t if timed else 0
                for a in range(A):          # ROS: push down actions sampled more often than the target says
                    if alpha > 0.0 and k > 0.0:
                        logits[a] = theta[ti, tt, s, a] - alpha * (C[s, a] - m[s] * prob[ti, tt, s, a]) / k
                    else:
                        logits[a] = theta[ti, tt, s, a]
                _log_softmax_into(logits, logp)
                a = sample_from_log_probs(logp, uniform(key, seed, j, t, SLOT_ACTION))
                C[s, a] += 1.0
                m[s] += 1.0
                k += 1.0
                s2, r = env_step(P_cum, R, reward_std, s, a, key, seed, j, t)
                N[s, a, s2] += 1.0
                RS[s, a, s2] += r
                RQ[s, a, s2] += r * r
                g += disc * r
                disc *= gamma
                s = s2
                steps += 1
            episodes += 1
            sum_returns += g
            j += 1
            while ck < step_checkpoints.shape[0] and steps >= step_checkpoints[ck]:
                est[0, i, ck] = sum_returns / episodes
                est[1, i, ck] = _mb_estimate(N, RS, starts, pi_e, terminal, horizon, gamma, V)
                ck += 1
        stats[0, i] = N
        stats[1, i] = RS
        stats[2, i] = RQ
        stat_starts[i] = starts


def run(mdp, pi_e_logits, target_logits, alpha=0.0, step_checkpoints=(10000,), n_seeds=1000, stream="default",
        time_indexed=False, init=None):
    """Estimates {"MC", "MB"} of shape (n_seeds, len(step_checkpoints)), and the final model statistics.

    target_logits: (S, A) or (n_seeds, S, A); with time_indexed, (L, S, A) or (n_seeds, L, S, A).
    init: statistics from an earlier run (its returned "stats"), pooled into the MB model."""
    step_checkpoints = np.asarray(step_checkpoints, np.int64)
    theta = np.asarray(target_logits, float)
    if not time_indexed:
        theta = theta[..., None, :, :]                       # a single time slice
    theta = np.ascontiguousarray(theta if theta.ndim == 4 else theta[None])
    assert theta.shape[0] in (1, n_seeds) and not (alpha > 0 and theta.shape[1] > 1)
    S, A = mdp.S, mdp.A
    est = np.zeros((2, n_seeds, len(step_checkpoints)))
    stats = np.zeros((3, n_seeds, S, A, S))
    stat_starts = np.zeros((n_seeds, S))
    if init is not None:
        init_arr = np.ascontiguousarray(np.stack([init[key] for key in STAT_KEYS[:3]]))
        init_starts = np.ascontiguousarray(init["starts"])
    else:
        init_arr, init_starts = np.zeros((3, 1, S, A, S)), np.zeros((1, S))
    _kernel(np.cumsum(mdp.P, axis=2), mdp.R, mdp.reward_std, mdp.terminal, np.cumsum(mdp.d0), int(mdp.horizon),
            float(mdp.gamma), exact.softmax(np.asarray(pi_e_logits, float)), theta, exact.softmax(theta),
            float(alpha), step_checkpoints, np.arange(n_seeds, dtype=np.int64), stream_key(stream), est,
            init_arr, init_starts, init is not None, stats, stat_starts)
    return {"MC": est[0], "MB": est[1],
            "stats": dict(zip(STAT_KEYS, (stats[0], stats[1], stats[2], stat_starts)))}
