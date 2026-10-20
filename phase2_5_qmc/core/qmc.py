"""Quasi-Monte Carlo (QMC) action sampling for policy evaluation.

Only the agent's action randomness is under our control, so QMC replaces the uniform that picks each action;
transitions and rewards stay i.i.d. Actions are drawn from a target policy q by inverse CDF of that uniform:
  * "per-state":  the k-th visit to state s uses u = frac(vdC(k) + shift_s), where vdC is the base-2
                  van der Corput sequence and shift_s ~ U[0, 1) per seed and state. Each state's action
                  frequencies then track q closely, like ROS but without adapting to the data.
  * "sobol":      trajectory j uses point j of a scrambled Sobol sequence, with coordinate t for step t.
                  Every trajectory is still exactly distributed as q, while whole action sequences are
                  spread out evenly across trajectories (the first 2^k points form a net).
  * "lhs":        the same with a Latin hypercube: each step's uniform is stratified across trajectories,
                  one coordinate at a time. Balanced only as a whole design, so use one checkpoint per run.
  * "antithetic": trajectories come in pairs whose action uniforms are u and 1 - u.
With alpha > 0, ROS's corrected probabilities are drawn with these uniforms instead of i.i.d. ones.

Estimators, as in Phase 2: MC (the mean return), B (OIS with q's probabilities) and R (OIS with the data's
empirical policy).
"""
import numpy as np
from numba import njit, prange
from scipy.stats import qmc

from phase1_reproduction.core import exact
from phase1_reproduction.core.rng import SLOT_START, env_step, sample_from_cdf, sample_from_log_probs, stream_key, uniform
from phase1_reproduction.core.sampling import _log_softmax_into

ESTIMATORS = ("MC", "B", "R")
SLOT_SHIFT = 5                       # random-number slot for the per-state shifts (slots 0-4 are taken)


@njit(cache=True)
def _van_der_corput(k):
    """Base-2 radical inverse of k: 0, 1/2, 1/4, 3/4, 1/8, ..."""
    x, f = 0.0, 0.5
    while k > 0:
        if k & 1:
            x += f
        k >>= 1
        f *= 0.5
    return x


@njit(parallel=True, cache=True)
def _kernel(P_cum, R, reward_std, terminal, d0_cum, horizon, gamma, log_pe, theta_q, log_q, q, alpha,
            per_state, points, checkpoints, seeds, key, est):
    S, A = log_pe.shape
    n_traj = checkpoints[-1]
    for i in prange(seeds.shape[0]):
        seed = seeds[i]
        C = np.zeros((S, A))                       # state-action counts (for ROS and for R)
        m = np.zeros(S)
        k = 0.0
        visits = np.zeros(S, np.int64)             # per-state visit index for van der Corput
        shift = np.empty(S)
        for s in range(S):
            shift[s] = uniform(key, seed, 0, s, SLOT_SHIFT)
        steps_s = np.empty(n_traj * horizon, np.int16)
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
        sum_b = 0.0
        for j in range(n_traj):
            s = sample_from_cdf(d0_cum, uniform(key, seed, j, 0, SLOT_START))
            g = 0.0
            disc = 1.0
            lw_e = 0.0
            lw_q = 0.0
            for t in range(horizon):
                if terminal[s]:
                    break
                for a in range(A):
                    if alpha > 0.0 and k > 0.0:
                        logits[a] = theta_q[s, a] - alpha * (C[s, a] - m[s] * q[s, a]) / k
                    else:
                        logits[a] = theta_q[s, a]
                _log_softmax_into(logits, logp)
                if per_state:
                    u = (_van_der_corput(visits[s]) + shift[s]) % 1.0
                    visits[s] += 1
                else:
                    u = points[i, j, t]
                a = sample_from_log_probs(logp, u)
                lw_e += log_pe[s, a]
                lw_q += log_q[s, a]
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
            sum_b += g * np.exp(lw_e - lw_q)
            n = j + 1
            if ck < checkpoints.shape[0] and n == checkpoints[ck]:
                est[0, i, ck] = sum_mc / n
                est[1, i, ck] = sum_b / n
                for ss in range(S):                # R: empirical policy of the first n trajectories
                    for aa in range(A):
                        log_pd[ss, aa] = np.log(C[ss, aa] / m[ss]) if C[ss, aa] > 0.0 else 0.0
                total = 0.0
                for jj in range(n):
                    lw_d = 0.0
                    for p in range(ends[jj], ends[jj + 1]):
                        lw_d += log_pd[steps_s[p], steps_a[p]]
                    total += returns[jj] * np.exp(log_pe_traj[jj] - lw_d)
                est[2, i, ck] = total / n
                ck += 1


def _trajectory_points(sampler, n, horizon, seed):
    """Action uniforms (n trajectories, horizon steps) for one seed."""
    if sampler == "sobol":
        return qmc.Sobol(d=horizon, scramble=True, seed=np.random.default_rng([20261019, seed])).random_base2(int(np.log2(n)))
    rng = np.random.default_rng([20261019, seed, 1])
    if sampler == "lhs":
        return qmc.LatinHypercube(d=horizon, seed=rng).random(n)
    u = rng.random((n // 2, horizon))                      # antithetic: rows 2i and 2i + 1 are u and 1 - u
    return np.stack([u, 1.0 - u], axis=1).reshape(n, horizon)


def run(mdp, pi_e_logits, target_logits, sampler, alpha=0.0, checkpoints=(1024,), n_seeds=1000, stream="default",
        chunk=25):
    """Estimates {MC, B, R} of shape (n_seeds, len(checkpoints)) from data drawn from the target with QMC uniforms.

    sampler: "per-state", "sobol", "lhs" or "antithetic". For the trajectory-level samplers the points are
    generated in chunks of seeds (each seed gets its own randomization) to bound memory."""
    checkpoints = np.asarray(checkpoints, np.int64)
    n_traj = int(checkpoints[-1])
    assert n_traj & (n_traj - 1) == 0, "the trajectory-level designs need a power-of-two number of trajectories"
    theta = np.ascontiguousarray(target_logits, dtype=float)
    est = np.zeros((len(ESTIMATORS), n_seeds, len(checkpoints)))
    args = (np.cumsum(mdp.P, axis=2), mdp.R, mdp.reward_std, mdp.terminal, np.cumsum(mdp.d0), int(mdp.horizon),
            float(mdp.gamma), exact.log_softmax(np.asarray(pi_e_logits, float)), theta, exact.log_softmax(theta),
            exact.softmax(theta), float(alpha))
    key = stream_key(stream)
    if sampler == "per-state":
        _kernel(*args, True, np.zeros((1, 1, 1)), checkpoints, np.arange(n_seeds, dtype=np.int64), key, est)
        return dict(zip(ESTIMATORS, est))
    for lo in range(0, n_seeds, chunk):
        seeds = np.arange(lo, min(lo + chunk, n_seeds), dtype=np.int64)
        points = np.stack([_trajectory_points(sampler, n_traj, mdp.horizon, s) for s in seeds])
        part = np.zeros((len(ESTIMATORS), len(seeds), len(checkpoints)))
        _kernel(*args, False, points, checkpoints, seeds, key, part)
        est[:, lo:lo + len(seeds)] = part
    return dict(zip(ESTIMATORS, est))
