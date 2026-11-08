"""CartPole-v0 with the ROS repository's evaluation policy: coordinated sampling in a continuous state space.

Dynamics: gym's CartPole-v0 (Euler steps of 0.02 s; the episode fails once the pole passes 12 degrees or the cart
passes 2.4; at most 200 steps; reward 1 per step; start state uniform on [-0.05, 0.05]^4). An optional Gaussian
noise on the push force (std `noise`, in newtons) makes the dynamics stochastic.

Policies: the repo's CartPole_1000 network (BatchNorm in eval mode, two ReLU layers of 64, softmax over push left /
push right). All trainable parameters sit in one flat vector theta, so ROS and BPG move them as one vector; the
BatchNorm statistics stay fixed, as in the repo.

Samplers (per-seed estimates: MC = mean return, OIS, PDIS; plus the mean episode length):
  * iid:   n independent episodes from a behavior network.
  * ros:   on-policy ROS as in the repo: after every step the behavior is theta_e - lr * (mean over all past steps
           of grad log pi_e(a|s)), which lowers the probability of actions taken more often than pi_e would.
  * array: Array-RQMC. The n episodes run in lockstep; at each step the running ones are sorted by a value estimate
           V(s) = w . [h(s), 1] (h = pi_e's last hidden layer), and the episode of rank i draws its action from
           u = frac(vdC(i) + shift_t). With block > 1, past weights are averaged over consecutive blocks of that
           many episodes in this order. That merge is exactly unbiased only if the episodes it averages have the
           same value-to-go under pi_e.
  * two_stage: the target is learned inside the budget. Online BPG as in the ROS repo (from pi_e, an SGD step on
           the OIS second moment after every 10 episodes) runs on the first n1 episodes; Array-RQMC with the learned
           target runs on the rest. The estimate is the episode-weighted mean of the two parts' OIS. With n1 = n it
           is online BPS with OIS, as in [1] and the repo.
bpg_target learns a fixed BPS target the same way, offline, with large batches.
"""
from collections import namedtuple

import numpy as np
from numba import njit, prange

from phase1_reproduction.core.repo_io import policy_tensors
from phase1_reproduction.core.rng import SLOT_ACTION, SLOT_NOISE_1, SLOT_NOISE_2, SLOT_START, stream_key, uniform
from phase2_5_qmc.core.array_rqmc import SLOT_STEP_SHIFT
from phase2_5_qmc.core.qmc import _van_der_corput

HORIZON, GAMMA = 200, 0.99
H = 64                                                  # hidden width
BN_W, BN_B, W1, B1, W2, B2, W3, B3, N_PARAMS = 0, 4, 8, 264, 328, 4424, 4488, 4616, 4618   # offsets in theta
THETA_MAX = 0.20943951023931953                         # 12 degrees
CHUNKS = 64                                             # work units for the reductions below
Net = namedtuple("Net", "theta mu inv_sd")              # parameters and the fixed BatchNorm statistics


def load_pi_e(name="CartPole_1000"):
    t = policy_tensors(name)
    theta = np.concatenate([t["model.0.0.weight"], t["model.0.0.bias"], t["model.0.1.weight"].ravel(),
                            t["model.0.1.bias"], t["model.0.3.weight"].ravel(), t["model.0.3.bias"],
                            t["model.1.0.weight"].ravel(), t["model.1.0.bias"]])
    assert theta.shape == (N_PARAMS,)
    return Net(theta, t["model.0.0.running_mean"], 1.0 / np.sqrt(t["model.0.0.running_var"] + 1e-5))


@njit(cache=True, fastmath=True)
def _forward(th, mu, inv_sd, s, x0, h1, h2):
    """Activations of the network at state s (written to x0, h1, h2); returns logit(right) - logit(left)."""
    for i in range(4):
        x0[i] = (s[i] - mu[i]) * inv_sd[i] * th[BN_W + i] + th[BN_B + i]
    for j in range(H):
        z = th[B1 + j]
        for i in range(4):
            z += th[W1 + 4 * j + i] * x0[i]
        h1[j] = max(z, 0.0)
    for j in range(H):
        z = th[B2 + j]
        for i in range(H):
            z += th[W2 + H * j + i] * h1[i]
        h2[j] = max(z, 0.0)
    z0, z1 = th[B3], th[B3 + 1]
    for i in range(H):
        z0 += th[W3 + i] * h2[i]
        z1 += th[W3 + H + i] * h2[i]
    return z1 - z0


@njit(cache=True)
def _log_p(d, a):
    """log pi(a | s) for the logit difference d = logit(right) - logit(left)."""
    x = d if a == 1 else -d
    return -np.log1p(np.exp(-x)) if x > 0.0 else x - np.log1p(np.exp(x))


@njit(cache=True)
def _action(d, u):
    """Inverse CDF with the actions in the order (left, right)."""
    return 1 if u >= 1.0 / (1.0 + np.exp(d)) else 0


@njit(cache=True, fastmath=True)
def _add_grad_logp(th, mu, inv_sd, s, a, x0, h1, h2, d, c, g, dh1, dx0):
    """g += c * grad_theta log pi(a | s), given the activations and output d of _forward(th, ..., s)."""
    delta = (1.0 if a == 1 else 0.0) - 1.0 / (1.0 + np.exp(-d))     # d log pi / d(logit right) = -d / d(logit left)
    g[B3] -= c * delta
    g[B3 + 1] += c * delta
    dh1[:] = 0.0
    dx0[:] = 0.0
    for j in range(H):                          # output layer, then back through the second ReLU layer
        g[W3 + j] -= c * delta * h2[j]
        g[W3 + H + j] += c * delta * h2[j]
        if h2[j] > 0.0:
            e = delta * (th[W3 + H + j] - th[W3 + j])
            g[B2 + j] += c * e
            for i in range(H):
                g[W2 + H * j + i] += c * e * h1[i]
                dh1[i] += th[W2 + H * j + i] * e
    for j in range(H):                          # first ReLU layer
        if h1[j] > 0.0:
            e = dh1[j]
            g[B1 + j] += c * e
            for i in range(4):
                g[W1 + 4 * j + i] += c * e * x0[i]
                dx0[i] += th[W1 + 4 * j + i] * e
    for i in range(4):                          # BatchNorm scale and shift
        g[BN_W + i] += c * dx0[i] * (s[i] - mu[i]) * inv_sd[i]
        g[BN_B + i] += c * dx0[i]


@njit(cache=True)
def _reset(s, key, seed, j):
    for i in range(4):
        s[i] = -0.05 + 0.1 * uniform(key, seed, j, i, SLOT_START)


@njit(cache=True)
def _step(s, a, noise, key, seed, j, t):
    """One CartPole-v0 step, in place; True when the episode fails."""
    force = 10.0 if a == 1 else -10.0
    if noise > 0.0:
        u1, u2 = uniform(key, seed, j, t, SLOT_NOISE_1), uniform(key, seed, j, t, SLOT_NOISE_2)
        force += noise * np.sqrt(-2.0 * np.log(1.0 - u1)) * np.cos(2.0 * np.pi * u2)
    x, x_dot, th, th_dot = s[0], s[1], s[2], s[3]
    cos, sin = np.cos(th), np.sin(th)
    temp = (force + 0.05 * th_dot * th_dot * sin) / 1.1
    th_acc = (9.8 * sin - cos * temp) / (0.5 * (4.0 / 3.0 - 0.1 * cos * cos / 1.1))
    x_acc = temp - 0.05 * th_acc * cos / 1.1
    s[0], s[1], s[2], s[3] = x + 0.02 * x_dot, x_dot + 0.02 * x_acc, th + 0.02 * th_dot, th_dot + 0.02 * th_acc
    return abs(s[0]) > 2.4 or abs(s[2]) > THETA_MAX


@njit(parallel=True, cache=True)
def _iid(th_e, th_b, on_policy, mu, inv_sd, noise, n, seeds, key, est):
    for i in prange(seeds.shape[0]):
        seed = seeds[i]
        s, x0, h1, h2 = np.empty(4), np.empty(4), np.empty(H), np.empty(H)
        tot = np.zeros(6)                       # sums of: return, return^2, OIS, OIS^2, PDIS, length
        for j in range(n):
            _reset(s, key, seed, j)
            g, disc, log_w, pdis, t = 0.0, 1.0, 0.0, 0.0, 0
            for t in range(HORIZON):
                d_e = _forward(th_e, mu, inv_sd, s, x0, h1, h2)
                d_b = d_e if on_policy else _forward(th_b, mu, inv_sd, s, x0, h1, h2)
                a = _action(d_b, uniform(key, seed, j, t, SLOT_ACTION))
                log_w += _log_p(d_e, a) - _log_p(d_b, a)
                done = _step(s, a, noise, key, seed, j, t)
                g += disc
                pdis += disc * np.exp(log_w)
                disc *= GAMMA
                if done:
                    break
            ois = g * np.exp(log_w)
            tot[0] += g
            tot[1] += g * g
            tot[2] += ois
            tot[3] += ois * ois
            tot[4] += pdis
            tot[5] += t + 1
        est[:, i] = tot / n


@njit(parallel=True, cache=True)
def _ros(th_e, mu, inv_sd, noise, lr, n, seeds, key, est):
    for i in prange(seeds.shape[0]):
        seed = seeds[i]
        s, x0, h1, h2, dh1, dx0 = np.empty(4), np.empty(4), np.empty(H), np.empty(H), np.empty(H), np.empty(4)
        grad = np.zeros(N_PARAMS)               # sum over past steps of grad log pi_e(a|s), taken at theta_e
        th_b = th_e.copy()
        steps, total = 0, 0.0
        for j in range(n):
            _reset(s, key, seed, j)
            g, disc = 0.0, 1.0
            for t in range(HORIZON):
                a = _action(_forward(th_b, mu, inv_sd, s, x0, h1, h2), uniform(key, seed, j, t, SLOT_ACTION))
                d_e = _forward(th_e, mu, inv_sd, s, x0, h1, h2)
                _add_grad_logp(th_e, mu, inv_sd, s, a, x0, h1, h2, d_e, 1.0, grad, dh1, dx0)
                steps += 1
                f = lr / steps
                for p in range(N_PARAMS):
                    th_b[p] = th_e[p] - f * grad[p]
                done = _step(s, a, noise, key, seed, j, t)
                g += disc
                disc *= GAMMA
                if done:
                    break
            total += g
        est[0, i] = total / n
        est[1, i] = steps / n


@njit(cache=True)
def _array_one(th_e, th_b, on_policy, mu, inv_sd, vw, noise, n, block, seed, key, offset, out):
    """One Array-RQMC run of n lockstep episodes (episode indices offset .. offset + n - 1 in the random stream).
    out = (MC, OIS, PDIS, mean episode length)."""
    S = np.empty((n, 4))
    x0, h1, h2 = np.empty(4), np.empty(H), np.empty(H)
    g, disc, log_w, pdis = np.zeros(n), np.ones(n), np.zeros(n), np.zeros(n)
    alive = np.ones(n, np.bool_)
    d_e, d_b, value = np.empty(n), np.empty(n), np.empty(n)
    run = np.empty(n, np.int64)
    steps = 0
    for j in range(n):
        _reset(S[j], key, seed, offset + j)
    for t in range(HORIZON):
        m = 0
        for j in range(n):                      # both networks and the sort key, at the running episodes
            if alive[j]:
                d_e[j] = _forward(th_e, mu, inv_sd, S[j], x0, h1, h2)
                v = vw[H]
                for c in range(H):
                    v += vw[c] * h2[c]
                value[m] = v
                run[m] = j
                m += 1
                d_b[j] = d_e[j] if on_policy else _forward(th_b, mu, inv_sd, S[j], x0, h1, h2)
        if m == 0:
            break
        steps += m
        order = run[:m][np.argsort(value[:m])]
        if block > 1:                           # merge past weights over consecutive blocks (log-mean-exp)
            for lo in range(0, m, block):
                hi = min(lo + block, m)
                top = log_w[order[lo]]
                for r in range(lo + 1, hi):
                    top = max(top, log_w[order[r]])
                acc = 0.0
                for r in range(lo, hi):
                    acc += np.exp(log_w[order[r]] - top)
                avg = top + np.log(acc / (hi - lo))
                for r in range(lo, hi):
                    log_w[order[r]] = avg
        shift = uniform(key, seed, 0, t, SLOT_STEP_SHIFT)
        for r in range(m):
            j = order[r]
            a = _action(d_b[j], (_van_der_corput(r) + shift) % 1.0)
            log_w[j] += _log_p(d_e[j], a) - _log_p(d_b[j], a)
            if _step(S[j], a, noise, key, seed, offset + j, t):
                alive[j] = False
            g[j] += disc[j]
            pdis[j] += disc[j] * np.exp(log_w[j])
            disc[j] *= GAMMA
    out[0] = g.mean()
    out[1] = np.nan if block > 1 else (g * np.exp(log_w)).mean()
    out[2] = pdis.mean()
    out[3] = steps / n


@njit(parallel=True, cache=True)
def _array(th_e, th_b, on_policy, mu, inv_sd, vw, noise, n, block, seeds, key, est):
    for i in prange(seeds.shape[0]):
        _array_one(th_e, th_b, on_policy, mu, inv_sd, vw, noise, n, block, seeds[i], key, 0, est[:, i])


@njit(cache=True)
def _bpg_one(th_e, th_b, mu, inv_sd, noise, seed, key, j0, b, grad):
    """Episodes j0 .. j0 + b - 1 from th_b: adds IS^2 * sum_t grad log q(a_t | s_t) to grad.
    Returns (sum of the episodes' IS estimates, number of steps)."""
    s, x0, h1, h2, dh1, dx0 = np.empty(4), np.empty(4), np.empty(H), np.empty(H), np.empty(H), np.empty(4)
    score = np.empty(N_PARAMS)
    is_sum, steps = 0.0, 0
    for j in range(j0, j0 + b):
        _reset(s, key, seed, j)
        score[:] = 0.0
        g, disc, log_w = 0.0, 1.0, 0.0
        for t in range(HORIZON):
            d_e = _forward(th_e, mu, inv_sd, s, x0, h1, h2)
            d_b = _forward(th_b, mu, inv_sd, s, x0, h1, h2)            # last, so the activations are q's
            a = _action(d_b, uniform(key, seed, j, t, SLOT_ACTION))
            _add_grad_logp(th_b, mu, inv_sd, s, a, x0, h1, h2, d_b, 1.0, score, dh1, dx0)
            log_w += _log_p(d_e, a) - _log_p(d_b, a)
            steps += 1
            done = _step(s, a, noise, key, seed, j, t)
            g += disc
            disc *= GAMMA
            if done:
                break
        w = g * np.exp(log_w)
        is_sum += w
        for p in range(N_PARAMS):
            grad[p] += w * w * score[p]
    return is_sum, steps


@njit(parallel=True, cache=True)
def _bpg_batch(th_e, th_b, mu, inv_sd, noise, n, seed, key, grad):
    """Per chunk of a batch of n episodes from th_b: the sum of IS^2 * sum_t grad log q(a_t | s_t)."""
    C = grad.shape[0]
    for c in prange(C):
        lo, hi = c * n // C, (c + 1) * n // C
        if hi > lo:
            _bpg_one(th_e, th_b, mu, inv_sd, noise, seed, key, lo, hi - lo, grad[c])


@njit(parallel=True, cache=True)
def _two_stage(th_e, mu, inv_sd, vw, noise, n, n1, batch, lr, seeds, key, est):
    for i in prange(seeds.shape[0]):
        seed = seeds[i]
        th_b = th_e.copy()
        grad = np.empty(N_PARAMS)
        is_sum, steps = 0.0, 0.0
        for j0 in range(0, n1, batch):          # stage 1: online BPG as in the ROS repo, an SGD step per batch
            b = min(batch, n1 - j0)
            grad[:] = 0.0
            w, k = _bpg_one(th_e, th_b, mu, inv_sd, noise, seed, key, j0, b, grad)
            is_sum += w
            steps += k
            th_b += (lr / b) * grad             # descent on the OIS second moment
        if n > n1:                              # stage 2: Array-RQMC with the learned target
            out = np.empty(4)
            _array_one(th_e, th_b, False, mu, inv_sd, vw, noise, n - n1, 1, seed, key, n1, out)
            is_sum += (n - n1) * out[1]
            steps += (n - n1) * out[3]
        est[0, i] = is_sum / n
        est[1, i] = steps / n


def _seeds(n_seeds, first=0):
    return np.arange(first, first + n_seeds, dtype=np.int64)


def iid(pi_e, theta_b, n, n_seeds, stream, noise=0.0, first_seed=0):
    """Per-seed means over n i.i.d. episodes of theta_b (None = on-policy): MC, G2, OIS, OIS2, PDIS, length."""
    on = theta_b is None
    est = np.zeros((6, n_seeds))
    _iid(pi_e.theta, pi_e.theta if on else theta_b, on, pi_e.mu, pi_e.inv_sd, float(noise), int(n),
         _seeds(n_seeds, first_seed), stream_key(stream), est)
    return dict(zip(("MC", "G2", "OIS", "OIS2", "PDIS", "length"), est))


def ros(pi_e, lr, n, n_seeds, stream, noise=0.0):
    """Per-seed ROS-MC estimates from n episodes: MC, length."""
    est = np.zeros((2, n_seeds))
    _ros(pi_e.theta, pi_e.mu, pi_e.inv_sd, float(noise), float(lr), int(n), _seeds(n_seeds), stream_key(stream), est)
    return dict(zip(("MC", "length"), est))


def array(pi_e, theta_b, value_w, n, n_seeds, stream, noise=0.0, block=1):
    """Per-seed Array-RQMC estimates from n lockstep episodes of theta_b (None = on-policy): MC, OIS, PDIS, length."""
    assert n & (n - 1) == 0, "the Hammersley set needs a power-of-two number of episodes"
    on = theta_b is None
    est = np.zeros((4, n_seeds))
    _array(pi_e.theta, pi_e.theta if on else theta_b, on, pi_e.mu, pi_e.inv_sd, np.asarray(value_w, float),
           float(noise), int(n), int(block), _seeds(n_seeds), stream_key(stream), est)
    return dict(zip(("MC", "OIS", "PDIS", "length"), est))


def two_stage(pi_e, value_w, n, n1, lr, n_seeds, stream, noise=0.0, batch=10):
    """Per-seed estimates with the target learned inside the budget: online BPG on the first n1 episodes (an SGD step
    of size lr every `batch` episodes), Array-RQMC on the remaining n - n1 (a power of two, or 0). Returns OIS,
    length."""
    assert n == n1 or (n - n1) & (n - n1 - 1) == 0
    est = np.zeros((2, n_seeds))
    _two_stage(pi_e.theta, pi_e.mu, pi_e.inv_sd, np.asarray(value_w, float), float(noise), int(n), int(n1),
               int(batch), float(lr), _seeds(n_seeds), stream_key(stream), est)
    return dict(zip(("OIS", "length"), est))


def fit_value(pi_e, noise=0.0, n=20000, stream="value-fit", ridge=1e-6):
    """Sort key: least-squares fit of pi_e's discounted return to go on [h(s), 1], from n on-policy episodes."""
    xx, xy = np.zeros((CHUNKS, H + 1, H + 1)), np.zeros((CHUNKS, H + 1))
    _value_moments(pi_e.theta, pi_e.mu, pi_e.inv_sd, float(noise), int(n), stream_key(stream), xx, xy)
    A = xx.sum(0)
    return np.linalg.solve(A + ridge * np.trace(A) * np.eye(H + 1), xy.sum(0))


@njit(parallel=True, cache=True)
def _value_moments(th_e, mu, inv_sd, noise, n, key, xx, xy):
    """Per chunk: sums of f f^T and f * (discounted return to go) over on-policy steps, f = [h(s), 1]."""
    for c in prange(xx.shape[0]):
        s, x0, h1, h2 = np.empty(4), np.empty(4), np.empty(H), np.empty(H)
        f = np.ones((HORIZON, H + 1))
        for j in range(c, n, xx.shape[0]):
            _reset(s, key, 0, j)
            steps = 0
            for t in range(HORIZON):
                d = _forward(th_e, mu, inv_sd, s, x0, h1, h2)
                f[t, :H] = h2
                steps = t + 1
                if _step(s, _action(d, uniform(key, 0, j, t, SLOT_ACTION)), noise, key, 0, j, t):
                    break
            G = 0.0
            for t in range(steps - 1, -1, -1):
                G = 1.0 + GAMMA * G
                for p in range(H + 1):
                    xy[c, p] += f[t, p] * G
                    for q in range(H + 1):
                        xx[c, p, q] += f[t, p] * f[t, q]


def bpg_target(pi_e, noise=0.0, iters=300, batch=2000, step=0.05, stream="bpg", n_val=20000, every=25):
    """A fixed BPS target, learned offline: from pi_e, unit-norm steps that lower the OIS second moment E_q[(g w)^2].
    Every `every` steps the second moment is measured on one fixed validation sample; the best iterate is returned.
    Uses iters * batch + (iters / every + 1) * 10 * n_val episodes."""
    key = stream_key(stream)

    def second_moment(theta):
        return iid(pi_e, theta, n_val, 10, stream + "-val", noise)["OIS2"].mean()

    theta = pi_e.theta.copy()
    best = (second_moment(theta), theta.copy())
    for k in range(iters):
        grad = np.zeros((CHUNKS, N_PARAMS))
        _bpg_batch(pi_e.theta, theta, pi_e.mu, pi_e.inv_sd, float(noise), int(batch), k, key, grad)
        direction = grad.sum(0)                 # = batch * (minus the gradient of the second moment)
        theta = theta + step / np.sqrt(1.0 + k / 50) * direction / np.linalg.norm(direction)
        if (k + 1) % every == 0:
            m2 = second_moment(theta)
            if m2 < best[0]:
                best = (m2, theta.copy())
    return best[1]
