"""BPS + ROS: learn a behavior policy with BPS ([1]), collect data by tracking it with ROS ([2]),
then reweight the data to estimate v(pi_e).

1. Target. The behavior policy q that minimizes the variance of ordinary importance sampling (OIS).
   It is either learned per seed by BPG-V from stage-1 trajectories (learned_bps_target) or computed
   exactly (oracle_bps_target).
2. Collection. Trajectories come from q, either i.i.d. (alpha = 0) or with ROS (alpha > 0), which
   steers each action so the empirical action frequencies of the data track q.
3. Estimation. OIS with w = prod_t pi_e(a_t | s_t) / b(a_t | s_t), for three choices of b:
     A: the probability each action was actually drawn with (ROS's corrected policy);
     B: the target q, as if the data were i.i.d. from q;
     R: the empirical policy of the data, pi_D(a | s) = C[s, a] / m[s] (regression IS).
   Plain Monte Carlo (MC) is also returned; it is unbiased only when q = pi_e.

Environments, exact DP, random numbers and BPG-V are reused from Phase 1.
"""
import numpy as np
from numba import njit, prange

from phase1_reproduction.core import bpg, exact
from phase1_reproduction.core.rng import (SLOT_ACTION, SLOT_START, env_step, sample_from_cdf,
                                          sample_from_log_probs, stream_key, uniform)
from phase1_reproduction.core.sampling import _log_softmax_into

ESTIMATORS = ("MC", "A", "B", "R")


def oracle_bps_target(mdp, pi_e_logits):
    """Logits of the exact OIS-variance-minimizing tabular behavior policy."""
    return exact.oracle_behavior(mdp, exact.softmax(pi_e_logits))[0]


def learned_bps_target(mdp, pi_e_logits, n_seeds, stream, n1=1024, k=10, step=0.1):
    """One BPS target per seed: the behavior policy BPG-V reaches after n1 stage-1 trajectories
    (batch k, unit-norm step, baseline). Returns logits of shape (n_seeds, S, A)."""
    out = bpg.run_bpg(mdp, pi_e_logits, k=k, step=step, normalize=True, baseline=True, checkpoints=[n1],
                      snap_iters=[n1 // k], n_seeds=n_seeds, stream=stream)
    return out["snapshots"][:, 0]


@njit(parallel=True, cache=True)
def _kernel(P_cum, R, reward_std, terminal, d0_cum, horizon, gamma, log_pe, theta_q, log_q, q, alpha,
            checkpoints, seeds, key, est):
    S, A = log_pe.shape
    n_traj = checkpoints[-1]
    for i in prange(seeds.shape[0]):
        seed = seeds[i]
        ti = np.int64(0)                       # index of this seed's target (one shared, or one per seed)
        if theta_q.shape[0] > 1:
            ti = np.int64(i)
        C = np.zeros((S, A))                   # state-action counts of the data so far
        m = np.zeros(S)
        k = 0.0
        steps_s = np.empty(n_traj * horizon, np.int16)   # every (s, a) visited, to recompute R's weights
        steps_a = np.empty(n_traj * horizon, np.int8)
        ends = np.zeros(n_traj + 1, np.int64)
        returns = np.empty(n_traj)
        log_pe_traj = np.empty(n_traj)
        logits = np.empty(A)
        logp = np.empty(A)
        log_pd = np.zeros((S, A))
        pos = 0
        ck = 0
        sum_mc = 0.0
        sum_a = 0.0
        sum_b = 0.0
        for j in range(n_traj):
            s = sample_from_cdf(d0_cum, uniform(key, seed, j, 0, SLOT_START))
            g = 0.0
            disc = 1.0
            lw_e = 0.0                         # log prod pi_e
            lw_drawn = 0.0                     # log prod of the probabilities actions were drawn with
            lw_q = 0.0                         # log prod q
            for t in range(horizon):
                if terminal[s]:
                    break
                for a in range(A):             # ROS: push down actions sampled more often than q says
                    if alpha > 0.0 and k > 0.0:
                        logits[a] = theta_q[ti, s, a] - alpha * (C[s, a] - m[s] * q[ti, s, a]) / k
                    else:
                        logits[a] = theta_q[ti, s, a]
                _log_softmax_into(logits, logp)
                a = sample_from_log_probs(logp, uniform(key, seed, j, t, SLOT_ACTION))
                lw_drawn += logp[a]
                lw_e += log_pe[s, a]
                lw_q += log_q[ti, s, a]
                C[s, a] += 1.0
                m[s] += 1.0
                k += 1.0
                steps_s[pos] = s
                steps_a[pos] = a
                pos += 1
                s, r = env_step(P_cum, R, reward_std, s, a, key, seed, j, t)
                g += disc * r
                disc *= gamma
            ends[j + 1] = pos
            returns[j] = g
            log_pe_traj[j] = lw_e
            sum_mc += g
            sum_a += g * np.exp(lw_e - lw_drawn)
            sum_b += g * np.exp(lw_e - lw_q)
            n = j + 1
            if ck < checkpoints.shape[0] and n == checkpoints[ck]:
                est[0, i, ck] = sum_mc / n
                est[1, i, ck] = sum_a / n
                est[2, i, ck] = sum_b / n
                # R: weights use the empirical policy of all n trajectories, so recompute them here
                for ss in range(S):
                    for aa in range(A):
                        log_pd[ss, aa] = np.log(C[ss, aa] / m[ss]) if C[ss, aa] > 0.0 else 0.0
                total = 0.0
                for jj in range(n):
                    lw_d = 0.0
                    for p in range(ends[jj], ends[jj + 1]):
                        lw_d += log_pd[steps_s[p], steps_a[p]]
                    total += returns[jj] * np.exp(log_pe_traj[jj] - lw_d)
                est[3, i, ck] = total / n
                ck += 1


def run(mdp, pi_e_logits, target_logits, alpha=0.0, checkpoints=(1024,), n_seeds=1000, stream="default"):
    """Collect from the target (i.i.d. if alpha = 0, else ROS) and estimate v(pi_e).

    target_logits: (S, A) shared by all seeds, or (n_seeds, S, A) with one target per seed.
    Returns {estimator: estimates of shape (n_seeds, len(checkpoints))} for MC, A, B and R."""
    checkpoints = np.asarray(checkpoints, np.int64)
    assert np.all(np.diff(checkpoints) > 0) and checkpoints[-1] < 2 ** 20 and n_seeds < 2 ** 16
    theta_q = np.asarray(target_logits, float)
    theta_q = np.ascontiguousarray(theta_q[None] if theta_q.ndim == 2 else theta_q)
    assert theta_q.shape[0] in (1, n_seeds)
    est = np.zeros((len(ESTIMATORS), n_seeds, len(checkpoints)))
    _kernel(np.cumsum(mdp.P, axis=2), mdp.R, mdp.reward_std, mdp.terminal, np.cumsum(mdp.d0), int(mdp.horizon),
            float(mdp.gamma), exact.log_softmax(np.asarray(pi_e_logits, float)), theta_q,
            exact.log_softmax(theta_q), exact.softmax(theta_q), float(alpha), checkpoints,
            np.arange(n_seeds, dtype=np.int64), stream_key(stream), est)
    return dict(zip(ESTIMATORS, est))
