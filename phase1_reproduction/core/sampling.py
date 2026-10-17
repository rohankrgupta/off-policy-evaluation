"""On-policy sampling (OS) and robust on-policy sampling (ROS, [2] Alg. 1), scored by Monte Carlo.

ROS counts the state-action pairs collected so far: C[s, a], m[s] = sum_a C[s, a], and k pairs in
total. Before every action it samples from softmax(theta_e - alpha * grad), where
grad = (C[s, :] - m[s] * pi_e(. | s)) / k is the gradient, at pi_e, of the average log-likelihood
of the data. Actions taken less often than pi_e prescribes become more likely, so the empirical
action frequencies track pi_e. This is the repo's RobustOnPolicySampler; alpha = 0 is OS.
"""
import numpy as np
from numba import njit, prange

from .exact import softmax
from .rng import SLOT_ACTION, SLOT_START, env_step, sample_from_cdf, sample_from_log_probs, stream_key, uniform


@njit(cache=True)
def _log_softmax_into(x, out):
    m = x[0]
    for i in range(1, x.shape[0]):
        m = max(m, x[i])
    z = 0.0
    for i in range(x.shape[0]):
        z += np.exp(x[i] - m)
    lse = m + np.log(z)
    for i in range(x.shape[0]):
        out[i] = x[i] - lse


@njit(parallel=True, cache=True)
def _kernel(P_cum, R, reward_std, terminal, d0_cum, horizon, gamma, theta, pi, alpha,
            checkpoints, seeds, key, estimates, returns, keep_returns):
    S, A = theta.shape
    n_traj = checkpoints[-1]
    for i in prange(seeds.shape[0]):
        seed = seeds[i]
        C = np.zeros((S, A))
        m = np.zeros(S)
        k = 0.0
        logits = np.empty(A)
        logp = np.empty(A)
        total = 0.0
        ck = 0
        for j in range(n_traj):
            s = sample_from_cdf(d0_cum, uniform(key, seed, j, 0, SLOT_START))
            g = 0.0
            disc = 1.0
            for t in range(horizon):
                if terminal[s]:
                    break
                for a in range(A):
                    if alpha > 0.0 and k > 0.0:
                        logits[a] = theta[s, a] - alpha * (C[s, a] - m[s] * pi[s, a]) / k
                    else:
                        logits[a] = theta[s, a]
                _log_softmax_into(logits, logp)
                a = sample_from_log_probs(logp, uniform(key, seed, j, t, SLOT_ACTION))
                C[s, a] += 1.0
                m[s] += 1.0
                k += 1.0
                s, r = env_step(P_cum, R, reward_std, s, a, key, seed, j, t)
                g += disc * r
                disc *= gamma
            total += g
            if keep_returns:
                returns[i, j] = g
            if ck < checkpoints.shape[0] and j + 1 == checkpoints[ck]:
                estimates[i, ck] = total / (j + 1)
                ck += 1


def run(mdp, pi_e_logits, alpha=0.0, checkpoints=(1024,), n_seeds=1000, stream="default", keep_returns=False):
    """Monte Carlo estimates of v(pi_e) from OS (alpha = 0) or ROS (alpha > 0) data.

    Each seed collects checkpoints[-1] trajectories. Returns estimates of shape
    (n_seeds, len(checkpoints)), the mean return of the first n trajectories at each checkpoint n.
    With keep_returns it also returns every trajectory's return, shape (n_seeds, checkpoints[-1])."""
    checkpoints = np.asarray(checkpoints, np.int64)
    assert np.all(np.diff(checkpoints) > 0) and checkpoints[-1] < 2 ** 20 and n_seeds < 2 ** 16
    theta = np.ascontiguousarray(pi_e_logits, dtype=float)
    estimates = np.zeros((n_seeds, checkpoints.shape[0]))
    returns = np.zeros((n_seeds, checkpoints[-1]) if keep_returns else (1, 1))
    _kernel(np.cumsum(mdp.P, axis=2), mdp.R, mdp.reward_std, mdp.terminal, np.cumsum(mdp.d0),
            int(mdp.horizon), float(mdp.gamma), theta, softmax(theta), float(alpha), checkpoints,
            np.arange(n_seeds, dtype=np.int64), stream_key(stream), estimates, returns, bool(keep_returns))
    return (estimates, returns) if keep_returns else estimates
