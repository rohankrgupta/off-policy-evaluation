"""Array-RQMC for the agent's actions (L'Ecuyer, Lécot & Tuffin 2008), tabular version.

All n episodes run in lockstep. At each step t the episodes are sorted by their current state (ties in random
order), and the episode of rank k draws its action with the uniform

    u = frac(vdC(k) + shift_t),

the second coordinate of a randomly shifted Hammersley point set (vdC = base-2 van der Corput; shift_t ~ U[0, 1)
fresh at every step). Episodes that share a state at the same step therefore get stratified actions.
Given the past, each uniform is exactly U[0, 1), so every trajectory is distributed as the sampling policy and
importance weights stay unbiased. Transitions use the same random numbers as the other samplers (episode k plays
the role of trajectory k), so comparisons share randomness.

Options for importance-weighted estimates (sampling policy q != pi_e):
  * by_weight: within a state, episodes are ordered by their current importance weight, so neighbours in the
    order (which get stratified uniforms) also carry similar weights.
  * merge: at every step, each episode's past weight is replaced by the average over the episodes in the same
    state. Futures from the same state are exchangeable, so this keeps the per-state sums of weights, and hence
    PDIS, unbiased, while removing weight differences within a state. (OIS, which also reweights past rewards,
    is no longer valid and is returned as NaN.)
  * action_order[t, s]: the order in which actions are laid out along [0, 1) when a uniform is turned into an
    action (default: 0, 1, ..., A - 1).
Estimates: MC (mean return), B (OIS with the sampling policy's probabilities) and PDIS (per-decision IS).
"""
import numpy as np
from numba import njit, prange

from phase1_reproduction.core import exact
from phase1_reproduction.core.rng import SLOT_START, env_step, sample_from_cdf, stream_key, uniform
from phase2_5_qmc.core.qmc import _van_der_corput

SLOT_PERM, SLOT_STEP_SHIFT = 6, 7      # random-number slots for tie-breaking and per-step shifts


@njit(cache=True)
def _sample_in_order(logp, order, u):
    """Action drawn with probabilities exp(logp), walking through the actions in the given order."""
    acc = 0.0
    last = order[0]
    for i in range(order.shape[0]):
        a = order[i]
        p = np.exp(logp[a])
        if p > 0.0:
            last = a
        acc += p
        if u < acc:
            return a
    return last


@njit(parallel=True, cache=True)
def _kernel(P_cum, R, reward_std, terminal, d0_cum, horizon, gamma, log_pe, log_q, n, by_weight, merge,
            action_order, seeds, key, est):
    S = log_pe.shape[0]
    for i in prange(seeds.shape[0]):
        seed = seeds[i]
        state = np.empty(n, np.int64)
        g = np.zeros(n)
        disc = np.ones(n)
        log_w = np.zeros(n)                        # log prod pi_e / q, for the OIS estimate
        pdis = np.zeros(n)                         # per-decision IS: sum_t gamma^t w_{0:t} r_t
        for k in range(n):
            state[k] = sample_from_cdf(d0_cum, uniform(key, seed, k, 0, SLOT_START))
        perm = np.empty(n, np.int64)
        order = np.empty(n, np.int64)
        start = np.zeros(S + 1, np.int64)
        for t in range(horizon):
            for k in range(n):                     # random order of episodes (Fisher-Yates) ...
                perm[k] = k
            for k in range(n - 1, 0, -1):
                r = np.int64(uniform(key, seed, k, t, SLOT_PERM) * (k + 1))
                perm[k], perm[r] = perm[r], perm[k]
            start[:] = 0                           # ... then a stable counting sort by state
            for k in range(n):
                start[state[k] + 1] += 1
            for s in range(S):
                start[s + 1] += start[s]
            for k in perm:
                order[start[state[k]]] = k
                start[state[k]] += 1
            lo = 0
            for s in range(S):                     # after placement, start[s] is the end of state s's block
                hi = start[s]
                if hi - lo > 1 and merge:          # replace past weights by the state's average (log-mean-exp)
                    top = log_w[order[lo]]
                    for r_ in range(lo + 1, hi):
                        top = max(top, log_w[order[r_]])
                    total = 0.0
                    for r_ in range(lo, hi):
                        total += np.exp(log_w[order[r_]] - top)
                    avg = top + np.log(total / (hi - lo))
                    for r_ in range(lo, hi):
                        log_w[order[r_]] = avg
                elif hi - lo > 1 and by_weight:    # within the state, order by current weight
                    block = order[lo:hi].copy()
                    order[lo:hi] = block[np.argsort(log_w[block], kind="mergesort")]
                lo = hi
            shift = uniform(key, seed, 0, t, SLOT_STEP_SHIFT)
            running = False
            for rank in range(n):
                k = order[rank]
                s = state[k]
                if terminal[s]:
                    continue
                running = True
                ao = action_order[t if action_order.shape[0] > 1 else 0, s]
                a = _sample_in_order(log_q[s], ao, (_van_der_corput(rank) + shift) % 1.0)
                log_w[k] += log_pe[s, a] - log_q[s, a]
                s2, r = env_step(P_cum, R, reward_std, s, a, key, seed, k, t)
                g[k] += disc[k] * r
                pdis[k] += disc[k] * np.exp(log_w[k]) * r
                disc[k] *= gamma
                state[k] = s2
            if not running:
                break
        est[0, i] = g.mean()
        est[1, i] = np.nan if merge else (g * np.exp(log_w)).mean()
        est[2, i] = pdis.mean()


def run(mdp, pi_e_logits, target_logits, n, n_seeds=1000, stream="default", by_weight=False, merge=False,
        action_order=None):
    """Estimates {"MC", "B", "PDIS"} of shape (n_seeds,) from n lockstep episodes per seed, drawn from the target."""
    assert n & (n - 1) == 0, "the Hammersley set needs a power-of-two number of episodes"
    est = np.zeros((3, n_seeds))
    if action_order is None:
        action_order = np.tile(np.arange(mdp.A), (1, mdp.S, 1))
    _kernel(np.cumsum(mdp.P, axis=2), mdp.R, mdp.reward_std, mdp.terminal, np.cumsum(mdp.d0), int(mdp.horizon),
            float(mdp.gamma), exact.log_softmax(np.asarray(pi_e_logits, float)),
            exact.log_softmax(np.asarray(target_logits, float)), int(n), bool(by_weight), bool(merge),
            np.ascontiguousarray(action_order, dtype=np.int64), np.arange(n_seeds, dtype=np.int64), stream_key(stream), est)
    return {"MC": est[0], "B": est[1], "PDIS": est[2]}
