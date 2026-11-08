"""Where to sample when the estimator is model-based (MB).

The MB estimator fits P(s'|s,a) and the mean reward from counts pooled over time, then evaluates
pi_e exactly in the fitted model. To first order (delta method), with n episodes from behavior b,

    n * Var(MB) = sum_{s,a} D_e(s,a)^2 sigma^2(s,a) / D_b(s,a),

where
  * D_pi(s,a) is the expected number of visits to (s, a) per episode under pi;
  * sigma^2(s,a) is the variance of the one-step target r + V(s') given (s, a): reward noise plus
    transition noise.
Action sampling does not appear: MB uses pi_e's probabilities exactly.

Two allocation rules are provided:
  * optimal_target: the stationary behavior minimizing this variance per episode, or per
    environment step (times the mean episode length). Found by L-BFGS with the exact gradient.
  * revar_target: ReVar's oracle allocation (Mukherjee, Hanna, Nowak 2022, Thm. 1) on the
    time-layered MDP, as ReVar applies it to DAGs. It assumes known transitions, so only reward
    variance enters.
Both can be learned from data (learned_target), by fitting a tabular model to stage-1 counts.
"""
import numpy as np
from scipy.optimize import minimize

from phase1_reproduction.core import exact
from phase1_reproduction.core.envs import TabularMDP


def visitation(mdp, b):
    """d[t, s] = Pr(S_t = s, episode still running) under b, stationary (S, A) or time-indexed (L, S, A)."""
    decide = ~mdp.terminal
    d = np.zeros((mdp.horizon, mdp.S))
    d[0] = mdp.d0 * decide
    for t in range(mdp.horizon - 1):
        bt = b[t] if b.ndim == 3 else b
        d[t + 1] = np.einsum("s,sa,sap->p", d[t], bt, mdp.P) * decide
    return d


def _bellman_noise(mdp, pi_e):
    """D_e(s,a) and sigma^2(s,a). V(s') is averaged over t with pi_e's visitation, since the model is pooled over t."""
    d = visitation(mdp, pi_e)
    V = exact._backward(mdp, pi_e)[1]                      # value-to-go V_t(s), shape (L + 1, S)
    De = d.sum(0)[:, None] * pi_e
    De[mdp.terminal] = 0.0
    total = d.sum(0, keepdims=True)
    w = np.divide(d, total, out=np.zeros_like(d), where=total > 0)
    Y = mdp.R + mdp.gamma * (w.T @ V[1:])[:, None, :]     # one-step target per (s, a, s')
    mean = (mdp.P * Y).sum(2, keepdims=True)
    sigma2 = (mdp.P * ((Y - mean) ** 2 + mdp.reward_std ** 2)).sum(2)
    return De, sigma2


def mb_variance(mdp, pi_e, b):
    """(n * Var(MB) per episode, mean episode length) when data come from b (stationary or time-indexed)."""
    De, sigma2 = _bellman_noise(mdp, pi_e)
    d = visitation(mdp, b)
    Db = np.einsum("ts,tsa->sa", d, b) if b.ndim == 3 else d.sum(0)[:, None] * b
    need = De ** 2 * sigma2 > 0
    var = np.inf if np.any(need & (Db <= 0)) else float((De[need] ** 2 * sigma2[need] / Db[need]).sum())
    return var, float(d.sum())


def per_step_ceiling(mdp, pi_e):
    """Upper bound on the per-step gain over on-policy sampling of any behavior:
    (sum D_e sigma^2)(sum D_e) / (sum D_e sigma)^2 = 1 + CV^2 of sigma(s,a) under pi_e's visits.
    By Cauchy-Schwarz this is the best allocation of a fixed number of steps if every (s, a) could get any
    share of visits; the MDP's flow constraints only lower the achievable gain."""
    De, sigma2 = _bellman_noise(mdp, pi_e)
    return float((De * sigma2).sum() * De.sum() / (De * np.sqrt(sigma2)).sum() ** 2)


def _policy_gradient(mdp, b, d, kappa):
    """Gradient w.r.t. softmax logits of sum_t sum_s d_t(s) kappa(s) (state 'reward' kappa)."""
    W = np.zeros((mdp.horizon + 1, mdp.S))
    for t in range(mdp.horizon - 1, -1, -1):
        W[t] = np.where(mdp.terminal, 0.0, kappa + (b * (mdp.P @ W[t + 1])).sum(1))
    grad = np.zeros_like(b)
    for t in range(mdp.horizon):
        Z = mdp.P @ W[t + 1]
        grad += d[t][:, None] * b * (Z - (b * Z).sum(1, keepdims=True))
    return grad


def optimal_target(mdp, pi_e, per_step=True):
    """Logits (S, A) of the stationary behavior minimizing the MB variance per step (or per episode)."""
    decide = ~mdp.terminal
    De, sigma2 = _bellman_noise(mdp, pi_e)
    weight = De ** 2 * sigma2
    theta0 = np.log(np.clip(pi_e, 1e-300, None))
    if not np.any(weight > 0):
        return theta0

    def objective(x):
        theta = theta0.copy()
        theta[decide] = x.reshape(-1, mdp.A)
        b = exact.softmax(theta)
        d = visitation(mdp, b)
        u = d.sum(0)                                       # expected visits per state
        Db = u[:, None] * b
        if np.any((weight > 0) & (Db <= 0)):
            return 1e12, np.zeros_like(x)
        ratio = np.divide(weight, Db, out=np.zeros_like(weight), where=weight > 0)
        F = ratio.sum()
        # d F: b appears directly in Db, and through the visits u (a policy gradient with kappa = dF/du)
        grad = -ratio + b * ratio.sum(1, keepdims=True)
        grad += _policy_gradient(mdp, b, d, np.divide(-ratio.sum(1), u, out=np.zeros(mdp.S), where=u > 0))
        f, g = np.log(F), grad / F                         # log objective: scale-free
        if per_step:                                       # times the mean episode length
            length = d.sum()
            f, g = f + np.log(length), g + _policy_gradient(mdp, b, d, decide.astype(float)) / length
        return f, g[decide].ravel()

    res = minimize(objective, theta0[decide].ravel(), jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    theta = theta0.copy()
    theta[decide] = res.x.reshape(-1, mdp.A)
    return theta


def revar_target(mdp, pi_e):
    """ReVar's oracle allocation, probabilities (L, S, A):
    b_t(a|s) ∝ sqrt(pi^2 [sigma_r^2(s,a) + gamma^2 sum_s' P(s'|s,a) B_{t+1}(s')^2]), B_t(s) = sum_a of the root.
    sigma_r^2(s,a) is the variance of the reward observed at (s, a)."""
    r_mean = (mdp.P * mdp.R).sum(2, keepdims=True)
    sigma_r2 = (mdp.P * ((mdp.R - r_mean) ** 2 + mdp.reward_std ** 2)).sum(2)
    B = np.zeros(mdp.S)
    b = np.zeros((mdp.horizon, mdp.S, mdp.A))
    for t in range(mdp.horizon - 1, -1, -1):
        root = np.sqrt(pi_e ** 2 * (sigma_r2 + mdp.gamma ** 2 * (mdp.P * B ** 2).sum(2)))
        total = root.sum(1, keepdims=True)
        b[t] = np.divide(root, total, out=pi_e.copy(), where=total > 0)
        B = np.where(mdp.terminal, 0.0, root.sum(1))
    return b


def fitted_mdp(template, stats):
    """Tabular MDP fitted to data: transition counts, reward sums and squared sums per (s, a, s'), start counts.
    Unvisited (s, a) borrow their state's pooled transitions; unvisited states are absorbing with reward 0."""
    counts, rsum, rsq = stats["counts"], stats["reward_sums"], stats["reward_sq_sums"]
    S, A = template.S, template.A
    P, R, var = np.zeros((S, A, S)), np.zeros((S, A, S)), np.zeros((S, A, S))
    for s in range(S):
        if template.terminal[s] or counts[s].sum() == 0:
            P[s, :, s] = 1.0
            continue
        for a in range(A):
            c, r1, r2 = (counts[s, a], rsum[s, a], rsq[s, a]) if counts[s, a].sum() > 0 else \
                (counts[s].sum(0), rsum[s].sum(0), rsq[s].sum(0))
            P[s, a] = c / c.sum()
            R[s, a] = np.divide(r1, c, out=np.zeros(S), where=c > 0)
            var[s, a] = np.maximum(np.divide(r2, c, out=np.zeros(S), where=c > 0) - R[s, a] ** 2, 0.0)
    d0 = stats["starts"] / stats["starts"].sum()
    return TabularMDP(template.name + " (fitted)", P, R, np.sqrt(var), template.terminal, d0, template.horizon,
                      template.gamma)


def learned_target(template, pi_e, stats, rule, mix=0.1):
    """A rule's allocation computed in the model fitted to one seed's stage-1 data, mixed with pi_e
    (weight `mix`) so no action pi_e takes is starved. Returns logits: (S, A) for "ours", (L, S, A) for "revar"."""
    fit = fitted_mdp(template, stats)
    b = exact.softmax(optimal_target(fit, pi_e)) if rule == "ours" else revar_target(fit, pi_e)
    return np.log(np.clip((1 - mix) * b + mix * pi_e, 1e-300, None))
