"""Behavior policy gradient (BPG-V, [1] Alg. 1) with the adaptive OIS estimate it produces.

Trajectories are sampled in batches of k from the current behavior policy pi_theta, a tabular
softmax started at pi_e. After each batch the logits move along an estimate of -grad Var_IS
([1] Thm. 1):

    g_hat = (1/k) sum_j (x_j^2 - b) sum_t grad log pi_theta(A_t | S_t),
    x_j   = (G_j / reward_scale) * prod_t pi_e(A_t | S_t) / pi_theta(A_t | S_t).

Two settings are used:
  * [1]'s BPG-V: baseline b = mean x^2 of the previous batch, unit-norm step
    theta += step * g_hat / ||g_hat||.
  * The ROS repo's BehaviorPolicyGradient: no baseline, plain SGD theta += step * g_hat,
    rewards divided by v (reward_scale = v), and a first batch of k - 1 trajectories
    (repo_off_by_one=True).

The value estimate after n trajectories is the mean of G_j * w_j over all of them. It is unbiased
even though the behavior changes ([1] Prop. 1).
"""
import numpy as np
from numba import njit, prange

from .exact import is_second_moment, log_softmax, softmax, value
from .rng import SLOT_ACTION, SLOT_START, env_step, sample_from_cdf, sample_from_log_probs, stream_key, uniform


@njit(cache=True)
def _policy_tables(theta, logp, prob):
    """Fill log-probabilities and probabilities of the softmax policy theta."""
    S, A = theta.shape
    for s in range(S):
        m = theta[s, 0]
        for a in range(1, A):
            m = max(m, theta[s, a])
        z = 0.0
        for a in range(A):
            z += np.exp(theta[s, a] - m)
        lz = m + np.log(z)
        for a in range(A):
            logp[s, a] = theta[s, a] - lz
            prob[s, a] = np.exp(logp[s, a])


@njit(parallel=True, cache=True)
def _kernel(P_cum, R, reward_std, terminal, d0_cum, horizon, gamma, log_pe, theta0, k, offset, step,
            normalize, use_baseline, reward_scale, checkpoints, snap_iters, seeds, key, estimates, snapshots):
    S, A = log_pe.shape
    n_traj = checkpoints[-1]
    for i in prange(seeds.shape[0]):
        seed = seeds[i]
        theta = theta0.copy()
        logp = np.empty((S, A))
        prob = np.empty((S, A))
        _policy_tables(theta, logp, prob)
        grad = np.zeros((S, A))          # sum over the current batch of (x^2 - b) * grad log pi
        visited_s = np.empty(horizon, np.int64)
        visited_a = np.empty(horizon, np.int64)
        baseline = 0.0
        batch_x2 = 0.0
        batch_n = 0
        iteration = 0
        sum_is = 0.0
        ck = 0
        sn = 0
        while sn < snap_iters.shape[0] and snap_iters[sn] == 0:
            snapshots[i, sn] = theta
            sn += 1
        for j in range(n_traj):
            s = sample_from_cdf(d0_cum, uniform(key, seed, j, 0, SLOT_START))
            g = 0.0
            disc = 1.0
            log_pe_traj = 0.0            # log prod pi_e(a_t | s_t)
            log_pb_traj = 0.0            # log prod pi_theta(a_t | s_t)
            n_steps = 0
            for t in range(horizon):
                if terminal[s]:
                    break
                a = sample_from_log_probs(logp[s], uniform(key, seed, j, t, SLOT_ACTION))
                log_pe_traj += log_pe[s, a]
                log_pb_traj += logp[s, a]
                visited_s[n_steps] = s
                visited_a[n_steps] = a
                n_steps += 1
                s, r = env_step(P_cum, R, reward_std, s, a, key, seed, j, t)
                g += disc * r
                disc *= gamma
            w = np.exp(log_pe_traj - log_pb_traj)
            sum_is += g * w
            x = g / reward_scale * w
            x2 = x * x
            coef = x2 - baseline if use_baseline else x2
            for q in range(n_steps):                         # coef * sum_t grad log pi(a_t | s_t)
                ss, aa = visited_s[q], visited_a[q]
                grad[ss, aa] += coef
                for b in range(A):
                    grad[ss, b] -= coef * prob[ss, b]
            batch_x2 += x2
            batch_n += 1
            if (j + 1 + offset) % k == 0:                    # end of a batch: update the behavior
                if step != 0.0:
                    scale = step / k
                    if normalize:
                        norm = np.sqrt((grad * grad).sum())
                        scale = step / norm if norm > 0.0 else 0.0
                    theta += scale * grad
                grad[:] = 0.0
                baseline = batch_x2 / batch_n
                batch_x2 = 0.0
                batch_n = 0
                iteration += 1
                _policy_tables(theta, logp, prob)
                while sn < snap_iters.shape[0] and snap_iters[sn] == iteration:
                    snapshots[i, sn] = theta
                    sn += 1
            if ck < checkpoints.shape[0] and j + 1 == checkpoints[ck]:
                estimates[i, ck] = sum_is / (j + 1)
                ck += 1


def run_bpg(mdp, pi_e_logits, theta0=None, k=100, step=0.1, normalize=True, baseline=True, reward_scale=1.0,
            repo_off_by_one=False, checkpoints=(1024,), snap_iters=(), n_seeds=200, stream="bpg"):
    """Run BPG per seed from theta0 (default: pi_e's logits).

    Returns a dict with
      "estimates": adaptive OIS estimates at each trajectory checkpoint, shape (n_seeds, n_checkpoints);
      "snapshots": behavior logits after each iteration in snap_iters (0 = start), shape (n_seeds, n_snaps, S, A)."""
    checkpoints = np.asarray(checkpoints, np.int64)
    snap_iters = np.asarray(sorted(snap_iters), np.int64)
    assert np.all(np.diff(checkpoints) > 0) and checkpoints[-1] < 2 ** 20 and n_seeds < 2 ** 16
    pi_e_logits = np.asarray(pi_e_logits, float)
    theta0 = pi_e_logits.copy() if theta0 is None else np.asarray(theta0, float).copy()
    estimates = np.zeros((n_seeds, checkpoints.shape[0]))
    snapshots = np.zeros((n_seeds, max(1, snap_iters.shape[0]), mdp.S, mdp.A))
    _kernel(np.cumsum(mdp.P, axis=2), mdp.R, mdp.reward_std, mdp.terminal, np.cumsum(mdp.d0),
            int(mdp.horizon), float(mdp.gamma), log_softmax(pi_e_logits), theta0, int(k),
            int(repo_off_by_one), float(step), bool(normalize), bool(baseline), float(reward_scale),
            checkpoints, snap_iters, np.arange(n_seeds, dtype=np.int64), stream_key(stream),
            estimates, snapshots)
    return {"estimates": estimates, "snapshots": snapshots[:, : snap_iters.shape[0]]}


def snapshot_variance(mdp, pi_e, snapshots):
    """Exact Var_IS of every snapshot behavior policy; shape snapshots.shape[:-2].

    If a policy's probabilities underflow to 0 on an action pi_e takes, its IS variance is
    unbounded and is reported as inf."""
    v2 = value(mdp, pi_e) ** 2
    flat = snapshots.reshape(-1, mdp.S, mdp.A)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        out = np.array([is_second_moment(mdp, pi_e, softmax(th)) for th in flat]) - v2
    out[~np.isfinite(out)] = np.inf
    return out.reshape(snapshots.shape[:-2])
