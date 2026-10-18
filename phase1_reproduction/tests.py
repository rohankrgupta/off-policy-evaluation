"""Sanity checks (about 5 s). They test that:
  * our environments match the ROS repository;
  * the exact DP matches simulation and full enumeration;
  * BPG-V reaches the exact optimum.

Run:    python -m phase1_reproduction.tests
"""
import numpy as np

from phase1_reproduction.core import bpg, envs, exact, sampling


def test_environments_match_repo():
    g = envs.gridworld()
    pe = exact.softmax(envs.repo_policy_logits("GridWorld_5000", g))
    # the repo's own estimates: value and Var(g) from 1e6 rollouts, and the mean episode length
    assert abs(exact.value(g, pe) - 2.924171) < 3 * np.sqrt(21.45 / 1e6)
    assert abs(exact.return_variance(g, pe) / 21.451485 - 1) < 0.01
    assert abs(exact.mean_length(g, pe) - 7.432377) < 0.01
    b = envs.multibandit()
    pb = exact.softmax(envs.repo_policy_logits("MultiBandit_5000", b))
    assert abs(exact.value(b, pb) - 0.784954) < 3 * np.sqrt(0.19 / 1e6)


def test_dp_matches_simulation():
    g = envs.gridworld(slip=0.1)                      # stochastic transitions exercise the sampler fully
    logits = envs.repo_policy_logits("GridWorld_5000", g)
    pe = exact.softmax(logits)
    _, returns = sampling.run(g, logits, checkpoints=[4096], n_seeds=1000, stream="test", keep_returns=True)
    assert abs(returns.mean() - exact.value(g, pe)) < 4 * returns.std() / np.sqrt(returns.size)
    assert abs(returns.var() / exact.return_variance(g, pe) - 1) < 0.02


def test_is_variance_matches_enumeration():
    g = envs.gridworld(slip=0.1, horizon=5)
    logits = envs.repo_policy_logits("GridWorld_5000", g)
    pe, pb = exact.softmax(logits), exact.softmax(logits + np.random.default_rng(0).normal(0, 0.5, logits.shape))
    total = 0.0                                       # E_b[IS^2] by visiting every trajectory

    def visit(s, t, prob, ratio, ret):
        nonlocal total
        if t == g.horizon or g.terminal[s]:
            total += prob * (ratio * ret) ** 2
            return
        for a in range(g.A):
            for s2 in np.nonzero(g.P[s, a])[0]:
                visit(s2, t + 1, prob * pb[s, a] * g.P[s, a, s2], ratio * pe[s, a] / pb[s, a], ret + g.R[s, a, s2])

    visit(0, 0, 1.0, 1.0, 0.0)
    assert abs(exact.is_second_moment(g, pe, pb) / total - 1) < 1e-10


def test_bpg_reaches_oracle():
    b = envs.multibandit()
    logits = envs.repo_policy_logits("MultiBandit_5000", b)
    pe = exact.softmax(logits)
    out = bpg.run_bpg(b, logits, k=100, step=0.1, checkpoints=[300000], snap_iters=[3000], n_seeds=50, stream="test")
    assert bpg.snapshot_variance(b, pe, out["snapshots"]).mean() / exact.oracle_behavior(b, pe)[1] < 1.03


if __name__ == "__main__":
    for test in (test_environments_match_repo, test_dp_matches_simulation, test_is_variance_matches_enumeration,
                 test_bpg_reaches_oracle):
        test()
        print("passed:", test.__name__)
