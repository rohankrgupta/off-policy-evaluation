"""Linear-quadratic (LQ) control: a smooth, continuous-action testbed with an exact value.

    x_{t+1} = A x_t + B u_t + w_t,   w_t ~ N(0, sigma_w^2 I),   x_0 fixed,   reward_t = -(x_t' Q x_t + r u_t^2),
    pi_e:  u = -K x + sigma * eps,   eps ~ N(0, 1)   (scalar action; K is the LQR gain).

The action is a smooth function of its uniform, u = -K x + sigma * Phi^{-1}(v), unlike the GridWorld's step
functions. Only eps is the agent's randomness; w is the environment's.

Samplers of the agent's eps (all share the same environment noise w per seed and trajectory):
  iid, Sobol (one scrambled Sobol point per trajectory), antithetic (eps and -eps), Array-RQMC (episodes
  sorted at each step by the exact value-to-go x' P_t x), ROS ([2]: the behavior's mean and log-std are pi_e's
  minus alpha times the average score of the data so far) and PROPS ([3]: KL-regularized clipped-surrogate
  updates with Adam).
"""
from dataclasses import dataclass

import numpy as np
from numba import njit, prange
from scipy.linalg import solve_discrete_are
from scipy.special import ndtri
from scipy.stats import qmc

from phase1_reproduction.core.rng import uniform
from phase2_5_qmc.core.qmc import _van_der_corput

SLOT_EPS = (1, 4)                       # uniforms for the agent's normal (Box-Muller)
SLOT_W = 8                              # environment noise uses slots 8 + 2i and 9 + 2i for coordinate i


@dataclass
class LQ:
    A: np.ndarray                       # (d, d)
    B: np.ndarray                       # (d,)
    K: np.ndarray                       # (d,)  pi_e's mean action is -K x
    r: float                            # action cost
    sigma: float                        # pi_e's action std
    sigma_w: float                      # environment noise std
    x0: np.ndarray
    horizon: int


def make_lq(d, horizon, sigma_w, sigma=0.3, r=0.1):
    """1-D: A = B = 1. d > 1: a fixed random system (seed 0). Q = I; K is the infinite-horizon LQR gain."""
    if d == 1:
        A, B = np.eye(1), np.ones(1)
    else:
        rng = np.random.default_rng(0)
        A, B = np.eye(d) + 0.1 * rng.normal(size=(d, d)), rng.normal(size=d)
    P = solve_discrete_are(A, B[:, None], np.eye(d), np.array([[r]]))
    K = (B @ P @ A) / (r + B @ P @ B)
    return LQ(A, B, K, r, sigma, sigma_w, np.ones(d), horizon)


def value_to_go(lq):
    """P_t and c_t with E[sum of costs from t | x_t = x] = x' P_t x + c_t, by backward recursion."""
    d, H = len(lq.x0), lq.horizon
    F = lq.A - np.outer(lq.B, lq.K)                       # closed-loop dynamics
    P, c = np.zeros((H + 1, d, d)), np.zeros(H + 1)
    for t in range(H - 1, -1, -1):
        P[t] = np.eye(d) + lq.r * np.outer(lq.K, lq.K) + F.T @ P[t + 1] @ F
        c[t] = c[t + 1] + lq.sigma ** 2 * (lq.r + lq.B @ P[t + 1] @ lq.B) + lq.sigma_w ** 2 * np.trace(P[t + 1])
    return P, c


def value(lq):
    P, c = value_to_go(lq)
    return -(lq.x0 @ P[0] @ lq.x0 + c[0])


@njit(cache=True)
def _normal(key, seed, j, t, s1, s2):
    u1, u2 = uniform(key, seed, j, t, s1), uniform(key, seed, j, t, s2)
    return np.sqrt(-2.0 * np.log(1.0 - u1)) * np.cos(2.0 * np.pi * u2)


@njit(cache=True)
def noise(key, seed, n, horizon, d):
    """The agent's iid normals eps (n, H) and the environment's noise directions (n, H, d), counter-based."""
    eps, w = np.empty((n, horizon)), np.empty((n, horizon, d))
    for j in range(n):
        for t in range(horizon):
            eps[j, t] = _normal(key, seed, j, t, 1, 4)
            for i in range(d):
                w[j, t, i] = _normal(key, seed, j, t, SLOT_W + 2 * i, SLOT_W + 2 * i + 1)
    return eps, w


def rollout(lq, eps, w):
    """Returns of n trajectories whose action noise is eps (n, H) and environment noise sigma_w * w."""
    x = np.tile(lq.x0, (len(eps), 1))
    g = np.zeros(len(eps))
    for t in range(lq.horizon):
        u = -x @ lq.K + lq.sigma * eps[:, t]
        g -= (x * x).sum(1) + lq.r * u * u
        x = x @ lq.A.T + np.outer(u, lq.B) + lq.sigma_w * w[:, t]
    return g


def sobol_eps(n, horizon, seed):
    v = qmc.Sobol(d=horizon, scramble=True, seed=np.random.default_rng([20261019, seed])).random_base2(int(np.log2(n)))
    return ndtri(np.clip(v, 1e-12, 1 - 1e-12))


def array_rqmc(lq, w, seed, key):
    """Array-RQMC returns: at each step, episodes sorted by x' P_t x get eps = Phi^{-1}(frac(vdC(rank) + shift_t))."""
    n = len(w)
    P, _ = value_to_go(lq)
    vdc = np.array([_van_der_corput(k) for k in range(n)])
    x = np.tile(lq.x0, (n, 1))
    g = np.zeros(n)
    for t in range(lq.horizon):
        order = np.argsort(np.einsum("nd,de,ne->n", x, P[t], x))
        eps = np.empty(n)
        eps[order] = ndtri(np.clip((vdc + uniform(key, seed, 0, t, 7)) % 1.0, 1e-12, 1 - 1e-12))
        u = -x @ lq.K + lq.sigma * eps
        g -= (x * x).sum(1) + lq.r * u * u
        x = x @ lq.A.T + np.outer(u, lq.B) + lq.sigma_w * w[:, t]
    return g


@njit(parallel=True, cache=True)
def _adaptive(A, B, K, r, sigma, sigma_w, x0, horizon, n, method, alpha, lr, seeds, key, out):
    """Sequential ROS (method 0) or PROPS (method 1), in parallel over seeds; method 0 with alpha = 0 is i.i.d.
    Behavior: u ~ N(mean_b, std_b^2), drawn as u = mean_b + std_b * eps_jt with the same counter-based normals as
    noise(). out[seed index, j] = return of trajectory j."""
    d = x0.shape[0]
    for si in prange(seeds.shape[0]):
        _one_seed(A, B, K, r, sigma, sigma_w, x0, horizon, n, method, alpha, lr, seeds[si], key, out[si], d)


@njit(cache=True)
def _one_seed(A, B, K, r, sigma, sigma_w, x0, horizon, n, method, alpha, lr, seed, key, out, d):
    total = n * horizon
    xs, us = np.empty((total, d)), np.empty(total)        # data so far (for PROPS)
    s_eps, s_epsx, s_sq, k = 0.0, np.zeros(d), 0.0, 0     # ROS: sums of pi_e's standardized residuals
    phi = np.zeros(d + 2)                                 # PROPS: behavior params (dK, db, dlogs) relative to pi_e
    np.random.seed(seed)                                  # PROPS minibatches
    for j in range(n):
        x = x0.copy()
        g = 0.0
        if method == 1 and j > 0 and j % 2 == 0:          # PROPS update every 2 episodes
            phi[:] = 0.0
            m1, m2 = np.zeros(d + 2), np.zeros(d + 2)
            for epoch in range(8):
                grad, kl = np.zeros(d + 2), 0.0
                for _ in range(256):                       # minibatch from all data so far
                    i = np.random.randint(k)
                    xi, ui = xs[i], us[i]
                    mu_e = -(K @ xi)
                    mu_p = mu_e - phi[:d] @ xi + phi[d]
                    s_p = sigma * np.exp(phi[d + 1])
                    z = (ui - mu_p) / s_p
                    ratio = np.exp(-0.5 * z * z - phi[d + 1] + 0.5 * ((ui - mu_e) / sigma) ** 2)
                    score = np.empty(d + 2)                # grad of log pi_phi(u|x) w.r.t. (dK, db, dlogs)
                    score[:d] = -z / s_p * xi
                    score[d] = z / s_p
                    score[d + 1] = z * z - 1.0
                    if ratio >= 1.0 - 0.3:                 # clipped surrogate: push down sampled actions
                        grad -= ratio * score
                    diff = mu_e - mu_p                     # KL(pi_e || pi_phi) and its gradient
                    kl += np.log(s_p / sigma) + (sigma * sigma + diff * diff) / (2 * s_p * s_p) - 0.5
                    gk = np.empty(d + 2)
                    gk[:d] = diff / (s_p * s_p) * xi
                    gk[d] = -diff / (s_p * s_p)
                    gk[d + 1] = 1.0 - (sigma * sigma + diff * diff) / (s_p * s_p)
                    grad -= 0.1 * gk
                grad /= 256.0
                if kl / 256.0 > 0.1:                       # KL cutoff
                    break
                m1 = 0.9 * m1 + 0.1 * grad                 # Adam ascent
                m2 = 0.999 * m2 + 0.001 * grad * grad
                phi += lr * (m1 / (1 - 0.9 ** (epoch + 1))) / (np.sqrt(m2 / (1 - 0.999 ** (epoch + 1))) + 1e-8)
        for t in range(horizon):
            mu_e = -(K @ x)
            if method == 0 and alpha > 0.0 and k > 0:     # ROS: theta_b = theta_e - alpha * mean score
                shift = -alpha * (s_epsx @ x) / (sigma * k) - alpha * s_eps / (sigma * k)
                mean = mu_e + min(3.0 * sigma, max(-3.0 * sigma, shift))   # capped so actions stay in pi_e's range
                std = sigma * np.exp(min(3.0, max(-3.0, -alpha * s_sq / k)))
            elif method == 1:
                mean = mu_e - phi[:d] @ x + phi[d]
                std = sigma * np.exp(phi[d + 1])
            else:
                mean, std = mu_e, sigma
            u = mean + std * _normal(key, seed, j, t, 1, 4)
            res = (u - mu_e) / sigma                       # pi_e's standardized residual of this action
            s_eps += res
            s_epsx += res * x
            s_sq += res * res - 1.0
            xs[k] = x
            us[k] = u
            k += 1
            g -= (x @ x) + r * u * u
            wt = np.empty(d)
            for i in range(d):
                wt[i] = _normal(key, seed, j, t, SLOT_W + 2 * i, SLOT_W + 2 * i + 1)
            x = A @ x + B * u + sigma_w * wt
        out[j] = g


def adaptive(lq, n, method, alpha=0.0, lr=0.0, n_seeds=1000, key=None):
    """Per-trajectory returns (n_seeds, n) under ROS ("ros"; alpha = 0 gives i.i.d. sampling) or PROPS ("props")."""
    out = np.empty((n_seeds, n))
    _adaptive(lq.A, lq.B, lq.K, lq.r, lq.sigma, lq.sigma_w, lq.x0, lq.horizon, int(n), {"ros": 0, "props": 1}[method],
              float(alpha), float(lr), np.arange(n_seeds, dtype=np.int64), key, out)
    return out
