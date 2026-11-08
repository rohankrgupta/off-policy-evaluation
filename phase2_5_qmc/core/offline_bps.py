"""Offline BPS: learn the behavior target from a fixed on-policy pilot (CartPole, discrete actions).

BPG [1] uses each episode for one gradient step. Here one on-policy pilot is collected once and q is fitted to it:
    (1/n) sum_i g_i^2 pi_e(tau_i)/q(tau_i) / mean(g^2)  +  lam * mean over pilot steps of KL(pi_e(.|s_t) || q(.|s_t)),
the pilot's importance-sampling estimate of q's OIS second moment (relative to pi_e's), plus a KL pull toward pi_e on
the visited states. Minimized by L-BFGS with the exact gradient from a replay of the pilot. Fitting only the output
layer (subset "last") with lam = 1-10 is the recipe that works; the full network overfits. The same kernel evaluates
the cross-entropy objective (ce=True), -|g_i| sum_t log q(a_t|s_t), BPG-KL's target in [1].
On CartPole: Var_q(OIS)/Var(g) 0.45 / 0.38 / 0.27 from pilots of 512 / 1024 / 2048 episodes (BPG with 3.2e6: 0.247).
"""
import numpy as np
from numba import njit, prange
from scipy.optimize import minimize

from phase1_reproduction.core.rng import SLOT_ACTION, stream_key, uniform
from phase2_5_qmc.core.cartpole import (B3, GAMMA, H, HORIZON, N_PARAMS, W3, _act, _add_grad_logp, _forward, _log_p, _reset,
                                        _step)

CHUNKS = 64


@njit(parallel=True, cache=True)
def _collect(th_e, mu, inv_sd, noise, n, seed, key, S, A, L, G):
    """n on-policy episodes: states S[j, t], actions A[j, t], lengths L[j], discounted returns G[j]."""
    for j in prange(n):
        s, x0, h1, h2 = np.empty(4), np.empty(4), np.empty(H), np.empty(H)
        _reset(s, key, seed, j)
        g, disc, t = 0.0, 1.0, 0
        for t in range(HORIZON):
            z0, z1 = _forward(th_e, mu, inv_sd, s, x0, h1, h2)
            a = _act(z0, z1, uniform(key, seed, j, t, SLOT_ACTION), False)
            S[j, t, :] = s
            A[j, t] = a
            done = _step(s, a, False, noise, key, seed, j, t)
            g += disc
            disc *= GAMMA
            if done:
                break
        L[j] = t + 1
        G[j] = g


@njit(parallel=True, cache=True)
def _objective(th_e, th, mu, inv_sd, S, A, L, G, m2, lam_step, ce, vals, grads):
    """Per chunk c: vals[c] = (sum of the per-episode terms, sum over steps of KL), grads[c] = gradient of
    (1/n) sum_i term_i + lam_step * sum_steps KL.  term_i = g_i^2 pi_e(tau_i)/q(tau_i) / m2 (IS objective), or with
    ce: term_i = -|g_i| sum_t log q(a_t|s_t) / m2 (cross-entropy objective; m2 then holds mean |g|)."""
    n = G.shape[0]
    for c in prange(CHUNKS):
        x0, h1, h2, dh1, dx0 = np.empty(4), np.empty(H), np.empty(H), np.empty(H), np.empty(4)
        score = np.empty(N_PARAMS)
        for j in range(c, n, CHUNKS):
            score[:] = 0.0
            lr, lq = 0.0, 0.0
            for t in range(L[j]):
                s, a = S[j, t], A[j, t]
                e0, e1 = _forward(th_e, mu, inv_sd, s, x0, h1, h2)
                z0, z1 = _forward(th, mu, inv_sd, s, x0, h1, h2)        # last, so the activations are q's
                lr += _log_p(e0, e1, a, False) - _log_p(z0, z1, a, False)
                lq += _log_p(z0, z1, a, False)
                _add_grad_logp(th, mu, inv_sd, s, a, False, x0, h1, h2, z0, z1, 1.0, score, dh1, dx0)
                if lam_step > 0.0:                                     # KL(pi_e || q) at s and its gradient
                    for b in range(2):
                        pb = np.exp(_log_p(e0, e1, float(b), False))
                        vals[c, 1] += pb * (_log_p(e0, e1, float(b), False) - _log_p(z0, z1, float(b), False))
                        _add_grad_logp(th, mu, inv_sd, s, float(b), False, x0, h1, h2, z0, z1, -lam_step * pb,
                                       grads[c], dh1, dx0)
            if ce:
                w = abs(G[j]) / m2
                vals[c, 0] -= w * lq
            else:
                w = G[j] * G[j] * np.exp(min(lr, 50.0)) / m2
                vals[c, 0] += w
            for p in range(N_PARAMS):
                grads[c, p] -= w / n * score[p]


def collect(pi_e, n, seed, stream="offline-pilot", noise=0.0):
    S, A, L, G = np.zeros((n, HORIZON, 4)), np.zeros((n, HORIZON)), np.zeros(n, np.int64), np.zeros(n)
    _collect(pi_e.theta, pi_e.mu, pi_e.inv_sd, float(noise), int(n), int(seed), stream_key(stream), S, A, L, G)
    return S, A, L, G


SUBSETS = {"full": np.arange(N_PARAMS), "last": np.arange(W3, N_PARAMS), "bias": np.arange(B3, N_PARAMS)}


def learn(pi_e, pilot, lam, subset="last", ce=False, maxiter=100):
    """Target logits minimizing the offline objective over the parameter subset; returns (theta, objective, its)."""
    S, A, L, G = pilot
    n, T = len(G), float(L.sum())
    m = float(np.abs(G).mean()) if ce else float((G * G).mean())
    th0 = pi_e.theta.copy()
    idx = SUBSETS[subset]

    def f(x):
        th = th0.copy()
        th[idx] = x
        vals, grads = np.zeros((CHUNKS, 2)), np.zeros((CHUNKS, N_PARAMS))
        _objective(pi_e.theta, th, pi_e.mu, pi_e.inv_sd, S, A, L, G, m, lam / T, ce, vals, grads)
        return vals[:, 0].sum() / n + lam * vals[:, 1].sum() / T, grads.sum(0)[idx]

    res = minimize(f, th0[idx], jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    th = th0.copy()
    th[idx] = res.x
    return th, res.fun, res.nit


def is_estimate(pi_e, th, pilot):
    """The pilot's IS estimate of q's relative second moment, (1/n) sum_i g_i^2 pi_e(tau_i)/q(tau_i) / mean(g^2)."""
    S, A, L, G = pilot
    vals, grads = np.zeros((CHUNKS, 2)), np.zeros((CHUNKS, N_PARAMS))
    _objective(pi_e.theta, th, pi_e.mu, pi_e.inv_sd, S, A, L, G, float((G * G).mean()), 0.0, False, vals, grads)
    return vals[:, 0].sum() / len(G)


def split(pilot, half):
    S, A, L, G = pilot
    sl = slice(0, len(G) // 2) if half == 0 else slice(len(G) // 2, len(G))
    return S[sl], A[sl], L[sl], G[sl]
