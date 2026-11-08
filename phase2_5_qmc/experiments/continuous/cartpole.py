"""Continuous states: coordinated BPS on CartPole, the ROS repo's continuous-state benchmark (core/cartpole.py).

Domains: CartPole-v0 with the repo's CartPole_1000 policy, without and with Gaussian force noise (std 3 N).
Ground truth: 3e7 on-policy episodes. n = 64 / 512 / 4096 episodes, 1000 seeds; hyperparameters are tuned at each n
on 500 other seeds.

Part A, BPS target given: a fixed target learned beforehand by BPG (bpg_target; its episodes are reported, not
counted), like the exact targets of the GridWorld experiments.
  OS-MC; ROS-MC; BPS + OIS and BPS + PDIS (i.i.d.); on-policy Array-RQMC (MC); BPS + Array-RQMC with OIS and with
  PDIS; BPS + Array-RQMC + PDIS with merged pairs (neighbours in value order).
Part B, every episode counted, including those that learn the target (BPG learning rate tuned):
  BPS online [1]: the repo's online BPG on all n episodes (an SGD step every 10 episodes), OIS over all of them.
    Its gradient uses rewards divided by the true value, as the repo does, so the repo's learning rates apply;
  BPS online, then Array-RQMC: BPG on the first n/2 episodes, Array-RQMC with the learned target on the rest.
The sort key (a linear value fit on 20000 on-policy episodes) is not counted; without merging it affects only the
variance. Reference rows (noise 0): the ROS repo's stored CartPole runs (200 seeds) for OS, ROS (lr 10) and its
online BPG (lr 5e-5) with OIS, at the same number of environment steps.

Reported: n·MSE/Var(g) per episode and at equal environment steps (on-policy MC = 1), bias/SE, coverage and width of
95% t-intervals from groups of 10 runs, wall-clock time per episode at n = 4096 (10 cores), paired MSE ratios.

Run:    python -m phase2_5_qmc.experiments.continuous.cartpole     (about 45 minutes)
Output: results/continuous/cartpole.md
"""
import time
from pathlib import Path

import numpy as np

from phase1_reproduction.core import repo_io, stats
from phase2_5_qmc.core import cartpole as cp
from phase2_5_qmc.experiments.array_rqmc import ci_stats
from phase2_5_qmc.experiments.gridworld import mark, md

OUT = Path(__file__).resolve().parents[2] / "results" / "continuous"
NS = (64, 512, 4096)
N_EVAL, N_TUNE = 1000, 500
NOISES = (0.0, 3.0)
LRS = (3.0, 10.0, 30.0)                 # ROS learning rates (the repo's best for CartPole is 10)
BPG_LRS = (5e-5, 1e-4, 2e-4, 4e-4, 8e-4) # online BPG learning rates in Part B, on the repo's reward scale
REPO = {"OS-MC (ROS repo, stored)": "OnPolicySampler1_MonteCarlo",
        "ROS-MC, lr 10 (ROS repo, stored)": "RobustOnPolicySampler2_MonteCarlo",
        "BPS online, OIS (ROS repo's BPG, stored)": "BehaviorPolicyGradient2_OrdinaryImportanceSampling"}
REPO_TRUE_VALUE = 36.045859992477716    # the repo stores estimates divided by its true value
PART_A = ("OS-MC", "ROS-MC", "BPS + OIS (i.i.d.)", "BPS + PDIS (i.i.d.)", "Array-RQMC, on-policy (MC)",
          "BPS + Array-RQMC, OIS", "BPS + Array-RQMC, PDIS", "BPS + Array-RQMC, PDIS, merged pairs")
PART_B = ("OS-MC", "ROS-MC", "BPS online, OIS [1]", "BPS online, then Array-RQMC, OIS", *REPO)
CONTRASTS = (("BPS + Array-RQMC, OIS", "BPS + OIS (i.i.d.)"), ("BPS + Array-RQMC, OIS", "ROS-MC"),
             ("BPS + Array-RQMC, PDIS", "BPS + PDIS (i.i.d.)"),
             ("BPS + Array-RQMC, PDIS, merged pairs", "BPS + Array-RQMC, PDIS"),
             ("Array-RQMC, on-policy (MC)", "OS-MC"),
             ("BPS online, then Array-RQMC, OIS", "BPS online, OIS [1]"),
             ("BPS online, then Array-RQMC, OIS", "ROS-MC"), ("BPS online, OIS [1]", "ROS-MC"))


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def run_domain(noise):
    pi_e = cp.load_pi_e()
    truth = cp.iid(pi_e, None, 1_000_000, 30, "truth", noise)
    v = truth["MC"].mean()
    var = truth["G2"].mean() - v * v
    L_e = truth["length"].mean()
    vw = cp.fit_value(pi_e, noise)
    t0 = time.perf_counter()
    th_b = cp.bpg_target(pi_e, noise)
    t_bpg = time.perf_counter() - t0
    chk = cp.iid(pi_e, th_b, 100_000, 10, "target-check", noise)
    D = {"noise": noise, "v": v, "se": truth["MC"].std(ddof=1) / np.sqrt(30), "var": var, "L_e": L_e,
         "t_bpg": t_bpg, "target_var": (chk["OIS2"].mean() - v * v) / var, "target_len": chk["length"].mean()}
    log(f"noise {noise}: v = {v:.4f} ± {D['se']:.4f}, Var(g) = {var:.1f}, BPS target {D['target_var']:.3f} "
        f"({t_bpg:.0f}s)")

    memo, methods = {}, {}

    def cached(tag, f):
        """f(n, n_seeds, stream, h) -> dict of per-seed arrays; runs once per arguments, with its run time."""
        def g(n, k, s, h):
            if (tag, n, k, s, h) not in memo:
                t = time.perf_counter()
                memo[(tag, n, k, s, h)] = (f(n, k, s, h), time.perf_counter() - t)
            return memo[(tag, n, k, s, h)]
        return g

    def evaluate(name, f, key, grid=(None,)):
        """Evaluation-seed estimates at every n; with a grid, h is the value with the lowest MSE on the tuning seeds."""
        est, length, chosen = [], [], []
        for n in NS:
            h = grid[0] if len(grid) == 1 else min(grid, key=lambda x: np.mean((f(n, N_TUNE, "tune", x)[0][key] - v) ** 2))
            r, sec = f(n, N_EVAL, "eval", h)
            est.append(r[key])
            length.append(r["length"].mean())
            chosen.append(h)
        methods[name] = {"est": np.stack(est, 1), "length": np.array(length), "chosen": chosen,
                         "us": 1e6 * sec / (N_EVAL * NS[-1]), "n": np.array(NS, float)}
        log(f"  {name}")

    os_ = cached("os", lambda n, k, s, h: cp.iid(pi_e, None, n, k, s, noise))
    bps = cached("bps", lambda n, k, s, h: cp.iid(pi_e, th_b, n, k, s, noise))
    arr = cached("arr", lambda n, k, s, h: cp.array(pi_e, th_b, vw, n, k, s, noise))
    evaluate("OS-MC", os_, "MC")
    evaluate("ROS-MC", cached("ros", lambda n, k, s, h: cp.ros(pi_e, h, n, k, s, noise)), "MC", LRS)
    evaluate("BPS + OIS (i.i.d.)", bps, "OIS")
    evaluate("BPS + PDIS (i.i.d.)", bps, "PDIS")
    evaluate("Array-RQMC, on-policy (MC)", cached("arr-on", lambda n, k, s, h: cp.array(pi_e, None, vw, n, k, s, noise)),
             "MC")
    evaluate("BPS + Array-RQMC, OIS", arr, "OIS")
    evaluate("BPS + Array-RQMC, PDIS", arr, "PDIS")
    evaluate("BPS + Array-RQMC, PDIS, merged pairs",
             cached("merge", lambda n, k, s, h: cp.array(pi_e, th_b, vw, n, k, s, noise, block=2)), "PDIS")
    evaluate("BPS online, OIS [1]", cached("online", lambda n, k, s, h: cp.two_stage(pi_e, vw, n, n, h / v ** 2, k, s, noise)),
             "OIS", BPG_LRS)
    evaluate("BPS online, then Array-RQMC, OIS",
             cached("online-arr", lambda n, k, s, h: cp.two_stage(pi_e, vw, n, n // 2, h / v ** 2, k, s, noise)), "OIS",
             BPG_LRS)
    if noise == 0.0:                            # the repo's stored runs: checkpoint j = 2^j episodes' worth of steps
        for name, file in REPO.items():
            est, steps = repo_io.stored_run("cartpole", f"CartPole_1000_{file}")
            idx = [int(np.log2(n)) for n in NS]
            methods[name] = {"est": est[:, idx] * REPO_TRUE_VALUE, "length": np.full(len(NS), L_e), "chosen": None,
                             "us": None, "n": steps[idx] / L_e}
    D["methods"] = methods
    return D


def report(results):
    text = ["# Coordinated BPS in a continuous state space: CartPole\n",
            "n·MSE/Var(g): on-policy Monte Carlo = 1, lower is better; 1000 seeds (stored repo rows: 200). "
            "'Equal steps' rescales by the method's mean episode length, i.e. compares at equal environment steps. "
            "Coverage: 95% t-intervals from groups of 10 runs. Time: µs per episode at n = 4096 on 10 cores.\n"]
    contrasts = []
    for D in results:
        M, v, var = D["methods"], D["v"], D["var"]
        text.append(f"## Force noise {D['noise']:g} N\n\nTruth v = {D['v']:.4f} ± {D['se']:.4f}, Var(g) = {var:.1f}, "
                    f"mean length {D['L_e']:.1f}. BPS target (Part A): Var_q(OIS)/Var(g) = {D['target_var']:.3f}, mean "
                    f"length {D['target_len']:.1f}; learned offline in {D['t_bpg']:.0f} s from 3.2e6 episodes (6e5 for the "
                    "gradient steps, 2.6e6 for validation; not counted in Part A).\n")
        width = {n: ci_stats(M["OS-MC"]["est"][:, j], v, 1.0)[1] for j, n in enumerate(NS)}
        for part, names in (("Part A: BPS target given", PART_A), ("Part B: every episode counted", PART_B)):
            rows = []
            for name in names:
                if name not in M:
                    continue
                m = M[name]
                y = m["n"] * ((m["est"] - v) ** 2).mean(0) / var
                e = m["est"][:, -1]
                cov, w = ci_stats(e, v, width[NS[-1]])
                chosen = "" if m["chosen"] is None or m["chosen"][0] is None else ", ".join(f"{c:g}" for c in m["chosen"])
                rows.append([name, *(f"{x:.3g}" for x in y), f"{y[-1] * m['length'][-1] / D['L_e']:.3g}",
                             f"{(e.mean() - v) / (e.std(ddof=1) / np.sqrt(len(e))):+.1f}", f"{cov:.0%}", f"{w:.2f}",
                             "–" if m["us"] is None else f"{m['us']:.1f}", chosen or "–"])
            text.append(f"### {part}\n\n" + md(["method", *(f"n = {n}" for n in NS), "equal steps, n = 4096",
                                                 "bias/SE", "coverage", "width", "µs/episode", "tuned (per n)"], rows)
                        + "\n")
        for x, y in CONTRASTS:
            cells = []
            for j in range(len(NS)):
                r, lo, hi = stats.paired_mse_ratio(M[x]["est"][:, j], M[y]["est"][:, j], v)
                cells.append(f"{r:.3g} {mark(lo, hi)}")
            contrasts.append([f"{D['noise']:g} N", f"{x} vs {y}", *cells])
    text.append("## Paired MSE ratios per episode\n\n✓ the first method is better, ✗ worse, ~ inconclusive (95% CI).\n\n"
                + md(["force noise", "comparison", *(f"n = {n}" for n in NS)], contrasts) + "\n")
    return "\n".join(text)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for noise in NOISES:
        results.append(run_domain(noise))
        (OUT / "cartpole.md").write_text(report(results))     # partial results survive an interruption
    print(report(results))


if __name__ == "__main__":
    main()
