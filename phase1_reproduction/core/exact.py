"""Exact quantities for finite-horizon tabular MDPs, by dynamic programming over (state, t).

Policies are stationary (S, A) probability tables. Write rho for the product of pi_e/pi_b ratios
from step t on, and G for the return from step t on. The core recursion runs backward and
computes E_b[rho^2], E_b[rho^2 G] and E_b[rho^2 G^2] for every (state, t), using only
K = pi_e^2 / pi_b.
  * With pi_b = pi_e it gives the on-policy value and return variance.
  * In general it gives the exact variance of the ordinary importance-sampling (OIS) estimate,
    which is the quantity behavior policy search (BPS) minimizes.
"""
import numpy as np
from scipy.optimize import minimize


def softmax(logits):
    z = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


def log_softmax(logits):
    m = logits.max(axis=-1, keepdims=True)
    return logits - m - np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))


def _transition_moments(mdp):
    """P, E[r 1{s'}] and E[r^2 1{s'}] per (s, a, s'), and the mask of decision states."""
    return mdp.P, mdp.P * mdp.R, mdp.P * (mdp.R ** 2 + mdp.reward_std ** 2), ~mdp.terminal


def _backward(mdp, K):
    """M0, M1, M2 of shape (horizon + 1, S): E_b[rho^2 G^k] from each state at each t (k = 0, 1, 2)."""
    P, PR, PR2, decide = _transition_moments(mdp)
    L, g = mdp.horizon, mdp.gamma
    M0, M1, M2 = np.ones((L + 1, mdp.S)), np.zeros((L + 1, mdp.S)), np.zeros((L + 1, mdp.S))
    for t in range(L - 1, -1, -1):
        m0, m1, m2 = M0[t + 1], M1[t + 1], M2[t + 1]
        E0 = P @ m0
        E1 = PR @ m0 + g * (P @ m1)
        E2 = PR2 @ m0 + 2 * g * (PR @ m1) + g * g * (P @ m2)
        M0[t] = np.where(decide, (K * E0).sum(1), 1.0)
        M1[t] = np.where(decide, (K * E1).sum(1), 0.0)
        M2[t] = np.where(decide, (K * E2).sum(1), 0.0)
    return M0, M1, M2


def _forward(mdp, K):
    """F0, F1, F2 of shape (horizon, S): E_b[rho_<t^2 G_<t^k 1{S_t = s, episode still running}],
    where rho_<t and G_<t cover the steps before t."""
    P, PR, PR2, decide = _transition_moments(mdp)
    L, g = mdp.horizon, mdp.gamma
    F0, F1, F2 = np.zeros((L, mdp.S)), np.zeros((L, mdp.S)), np.zeros((L, mdp.S))
    F0[0] = mdp.d0 * decide
    KP, KPR, KPR2 = K[:, :, None] * P, K[:, :, None] * PR, K[:, :, None] * PR2
    for t in range(L - 1):
        gt = g ** t
        f0, f1, f2 = F0[t], F1[t], F2[t]
        F0[t + 1] = np.einsum("sap,s->p", KP, f0)
        F1[t + 1] = np.einsum("sap,s->p", KP, f1) + gt * np.einsum("sap,s->p", KPR, f0)
        F2[t + 1] = (np.einsum("sap,s->p", KP, f2) + 2 * gt * np.einsum("sap,s->p", KPR, f1)
                     + gt * gt * np.einsum("sap,s->p", KPR2, f0))
        F0[t + 1, ~decide] = F1[t + 1, ~decide] = F2[t + 1, ~decide] = 0.0
    return F0, F1, F2


def value(mdp, pi):
    """v(pi)."""
    return float(mdp.d0 @ _backward(mdp, pi)[1][0])


def return_variance(mdp, pi):
    """Var(g) under pi: n times the MSE of on-policy Monte Carlo with n trajectories."""
    return float(mdp.d0 @ _backward(mdp, pi)[2][0]) - value(mdp, pi) ** 2


def is_second_moment(mdp, pi_e, pi_b):
    """E_b[(g * prod_t pi_e / pi_b)^2] (infinite if pi_b misses an action pi_e takes)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        K = np.where(pi_b > 0, pi_e ** 2 / pi_b, np.where(pi_e > 0, np.inf, 0.0))
    return float(mdp.d0 @ _backward(mdp, K)[2][0])


def is_variance(mdp, pi_e, pi_b):
    """Variance of one OIS term under pi_b: n times the MSE of OIS with n i.i.d. trajectories."""
    return is_second_moment(mdp, pi_e, pi_b) - value(mdp, pi_e) ** 2


def mean_length(mdp, pi):
    """Expected number of decisions per episode under pi."""
    return float(_forward(mdp, pi)[0].sum())


def is_second_moment_grad(mdp, pi_e, theta_b):
    """E_b[IS^2] and its exact gradient with respect to the softmax logits theta_b of pi_b.

    Uses d/dtheta E_b[IS^2] = -E_b[IS^2 sum_t grad log pi_b(A_t | S_t)] ([1], Thm. 1). Here
    W[t, s, a] = E_b[IS^2 1{S_t = s, A_t = a}] combines the forward and backward moments."""
    P, PR, PR2, decide = _transition_moments(mdp)
    L, g = mdp.horizon, mdp.gamma
    pi_b = softmax(theta_b)
    K = pi_e ** 2 / pi_b
    M0, M1, M2 = _backward(mdp, K)
    F0, F1, F2 = _forward(mdp, K)
    grad = np.zeros_like(pi_b)
    for t in range(L):
        gt = g ** t
        m0, m1, m2 = M0[t + 1], M1[t + 1], M2[t + 1]
        f0, f1, f2 = F0[t][:, None], F1[t][:, None], F2[t][:, None]
        # IS^2 = rho^2 (G_<t + g^t r_t + g^(t+1) G_>t)^2: expand the square, using past moments (F),
        # the reward at t (PR, PR2) and future moments (M).
        W = K * (f2 * (P @ m0) + 2 * gt * f1 * (PR @ m0) + gt * gt * f0 * (PR2 @ m0)
                 + 2 * g * gt * (f1 * (P @ m1) + gt * f0 * (PR @ m1)) + (g * gt) ** 2 * f0 * (P @ m2))
        W[~decide] = 0.0
        grad -= W - pi_b * W.sum(1, keepdims=True)
    return float(mdp.d0 @ M2[0]), grad


def oracle_behavior(mdp, pi_e, maxiter=5000):
    """The best tabular softmax behavior policy for OIS: argmin over theta of Var_IS(softmax(theta)).

    The objective is convex in the logits ([1], Thm. 2), so L-BFGS with the exact gradient finds the
    global optimum. Returns (logits, Var_IS at the optimum)."""
    decide = ~mdp.terminal
    theta0 = np.log(np.clip(pi_e, 1e-300, None))
    scale = is_second_moment(mdp, pi_e, pi_e)          # keeps the objective near 1

    def objective(x):
        theta = theta0.copy()
        theta[decide] = x.reshape(-1, mdp.A)
        m2, grad = is_second_moment_grad(mdp, pi_e, theta)
        return m2 / scale, grad[decide].ravel() / scale

    res = minimize(objective, theta0[decide].ravel(), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-10, "maxcor": 30})
    theta = theta0.copy()
    theta[decide] = res.x.reshape(-1, mdp.A)
    return theta, is_variance(mdp, pi_e, softmax(theta))
