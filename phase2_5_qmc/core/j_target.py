"""A behavior policy suited to coordinated sampling (Array-RQMC) with importance weights.

Under i.i.d. sampling, BPS minimizes the variance of the OIS estimate, E_q[(g w)^2] - v^2. Coordinated sampling
removes, to first order, the variance from which action is drawn in each state. What it cannot remove is the
reward and transition noise, weighted by squared importance weights:

    J(q) = sum_t E_q[ gamma^{2t} w_{0:t}^2 sigma_t^2(S_t, A_t) ],   sigma_t^2(s,a) = Var(r + gamma V_{t+1}(s') | s, a),

with w_{0:t} = prod_{k<=t} pi_e/q (these are the transition terms of the estimate's Doob martingale; for q = pi_e,
J is the Bellman-noise share that ROS-MC converges to). In a bandit, J = sum_a pi^2 sigma^2 / q, minimized by the
Neyman allocation q ∝ pi * sigma. J is convex in q's logits; the exact gradient comes from a backward pass
B_t(s) = sum_a K(s,a) [c_t(s,a) + sum_s' P B_{t+1}(s')], K = pi_e^2 / q, and the forward weights of exact._forward.
"""
import numpy as np
from scipy.optimize import minimize

from phase1_reproduction.core import exact


def step_noise(mdp, pi_e):
    """c[t, s, a] = gamma^{2t} Var(r + gamma V_{t+1}(s') | s, a) under pi_e's value-to-go V."""
    V = exact._backward(mdp, pi_e)[1]                      # (L + 1, S)
    c = np.zeros((mdp.horizon, mdp.S, mdp.A))
    for t in range(mdp.horizon):
        Y = mdp.R + mdp.gamma * V[t + 1][None, None, :]
        mean = (mdp.P * Y).sum(2, keepdims=True)
        c[t] = mdp.gamma ** (2 * t) * (mdp.P * ((Y - mean) ** 2 + mdp.reward_std ** 2)).sum(2)
    c[:, mdp.terminal] = 0.0
    return c


def objective_and_grad(mdp, pi_e, theta, c):
    """J(q) and its gradient with respect to the softmax logits theta of q."""
    q = exact.softmax(theta)
    K = pi_e ** 2 / q
    decide = ~mdp.terminal
    B = np.zeros((mdp.horizon + 1, mdp.S))
    G = np.zeros((mdp.horizon, mdp.S, mdp.A))              # K * (c_t + P B_{t+1}): per-action backward terms
    for t in range(mdp.horizon - 1, -1, -1):
        G[t] = K * (c[t] + mdp.P @ B[t + 1])
        B[t] = np.where(decide, G[t].sum(1), 0.0)
    F = exact._forward(mdp, K)[0]                          # F[t, s] = E_q[w_{0:t-1}^2 1{S_t = s}]
    grad = -np.einsum("ts,tsa->sa", F, G - q[None] * G.sum(2, keepdims=True))
    grad[~decide] = 0.0
    return float(mdp.d0 @ B[0]), grad


def optimal_target(mdp, pi_e, maxiter=5000):
    """Logits of the behavior minimizing J (L-BFGS with the exact gradient, started at pi_e)."""
    c = step_noise(mdp, pi_e)
    decide = ~mdp.terminal
    theta0 = np.log(np.clip(pi_e, 1e-300, None))
    scale = objective_and_grad(mdp, pi_e, theta0, c)[0]

    def f(x):
        theta = theta0.copy()
        theta[decide] = x.reshape(-1, mdp.A)
        J, g = objective_and_grad(mdp, pi_e, theta, c)
        return J / scale, g[decide].ravel() / scale

    res = minimize(f, theta0[decide].ravel(), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-10})
    theta = theta0.copy()
    theta[decide] = res.x.reshape(-1, mdp.A)
    return theta


def J(mdp, pi_e, theta):
    return objective_and_grad(mdp, pi_e, theta, step_noise(mdp, pi_e))[0]
