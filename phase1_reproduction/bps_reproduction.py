"""Reproduce BPS ([1]) on the ROS repository's GridWorld, and explain [2]'s BPS baseline.

Evaluation policies:
  * the repo's GridWorld_5000 (pi_e-A);
  * the repo's epsilon-greedy policies around it, with epsilon = 0.5, 0.6, 0.7 (pi_e-B is 0.7).

Part 1, [1]'s protocol. BPG-V with batch 100, unit-norm step 0.1 and a baseline runs 1000
iterations (1e5 trajectories) from pi_e, over 200 seeds. Everything is scored with exact variances:
  * the best possible variance reduction, 1 - Var_IS(oracle) / Var(g), where the oracle is the
    optimal tabular behavior policy;
  * [1]'s final-policy statistic, 1 - Var_IS(theta_1000) / Var(g);
  * the MSE of the adaptive OIS estimate, relative to on-policy Monte Carlo (exactly Var(g) / n).

Part 2, why BPS looked no better than on-policy sampling in [2]. Both BPG settings are run at [2]'s
budget of 8192 trajectories on pi_e-A and pi_e-B, over 1000 seeds.

Run:    python -m phase1_reproduction.bps_reproduction     (about 40 s)
Output: results/bps_reproduction.md, results/bps_reproduction.png
"""
from pathlib import Path

import matplotlib
import numpy as np

from phase1_reproduction.core import bpg, envs, exact, stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / "results"
SNAPS = (0, 1, 3, 10, 30, 100, 300, 1000)        # iterations at which the behavior policy is scored
CHECKPOINTS = (100, 1000, 10000, 100000)          # trajectories


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def part1(g, policies, axes):
    rows_policy, rows_mse = [], []
    for name, logits in policies.items():
        pe = exact.softmax(logits)
        var = exact.return_variance(g, pe)
        best = 1 - exact.oracle_behavior(g, pe)[1] / var
        out = bpg.run_bpg(g, logits, k=100, step=0.1, normalize=True, baseline=True, checkpoints=CHECKPOINTS,
                          snap_iters=SNAPS, n_seeds=200, stream="e1-bpg")
        var_ratio = bpg.snapshot_variance(g, pe, out["snapshots"]) / var       # (seeds, snapshots)
        final = var_ratio[:, -1]
        lo, hi = stats.bootstrap_ci(final)
        reduction = 1 - final.mean()
        rows_policy.append([name, f"{var:.1f}", f"{best:.1%}", f"{reduction:.1%} [{1 - hi:.1%}, {1 - lo:.1%}]",
                            f"{reduction / best:.0%}"])
        sq = (out["estimates"] - exact.value(g, pe)) ** 2
        os_mse = var / np.array(CHECKPOINTS)                                    # exact MSE of OS-MC
        ratio, se = sq.mean(0) / os_mse, sq.std(0, ddof=1) / np.sqrt(len(sq)) / os_mse
        rows_mse.append([name] + [f"{r:.2f} ± {s:.2f}" for r, s in zip(ratio, se)])
        line = axes[0].plot(np.array(SNAPS) + 1, np.median(var_ratio, 0), marker="o", ms=3, label=name)[0]
        axes[0].axhline(1 - best, color=line.get_color(), ls=":", lw=1)
        axes[1].errorbar(CHECKPOINTS, ratio, yerr=se, marker="o", ms=3, capsize=2, color=line.get_color())
        print(name, rows_policy[-1][2:], flush=True)
    return rows_policy, rows_mse


def part2(g, policies):
    rows = []
    settings = {   # [2]'s BPG is the repo's BehaviorPolicyGradient; [1]'s is BPG-V
        "[2]: batch 10, SGD lr 0.01, rewards / v": dict(k=10, step=0.01, normalize=False, baseline=False,
                                                         repo_off_by_one=True, snap_iters=[819]),
        "[1]: batch 100, unit-norm step 0.1, baseline": dict(k=100, step=0.1, normalize=True, baseline=True,
                                                              snap_iters=[81]),
    }
    for name in ("πe-A", "πe-B"):
        logits = next(lg for nm, lg in policies.items() if nm.startswith(name))
        pe = exact.softmax(logits)
        v, var = exact.value(g, pe), exact.return_variance(g, pe)
        for setting, kw in settings.items():
            scale = abs(v) if setting.startswith("[2]") else 1.0
            out = bpg.run_bpg(g, logits, reward_scale=scale, checkpoints=[8192], n_seeds=1000, stream="e1-r3", **kw)
            sq = (out["estimates"][:, 0] - v) ** 2 / (var / 8192)                # relative to OS-MC's exact MSE
            lo, hi = stats.bootstrap_ci(sq)
            final = bpg.snapshot_variance(g, pe, out["snapshots"])[:, 0] / var
            rows.append([name, setting, f"{sq.mean():.2f} [{lo:.2f}, {hi:.2f}]", f"{np.median(final):.3f}",
                         f"{np.mean(final > 2):.1%}"])
            print(rows[-1], flush=True)
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    g = envs.gridworld()
    base = envs.repo_policy_logits("GridWorld_5000", g)
    policies = {"πe-A (GridWorld_5000)": base}
    for eps in (0.5, 0.6, 0.7):
        policies[f"ε-greedy {eps}" if eps < 0.7 else "πe-B (ε-greedy 0.7)"] = envs.eps_greedy_logits(base, eps, g.terminal)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    rows_policy, rows_mse = part1(g, policies, axes)
    rows_baseline = part2(g, policies)
    text = (
        "# BPS reproduction: BPG-V on the GridWorld\n\n"
        "## 1. Variance reduction of the learned behavior policy ([1]'s protocol)\n\n"
        "BPG-V: batch 100, unit-norm step 0.1, baseline, 1000 iterations (10⁵ trajectories), 200 seeds. "
        "Reduction = 1 − Var_IS(θ)/Var(g), the MSE drop of fresh trajectories from behavior policy θ versus "
        "on-policy ones (exact). [1] reports 73.5% and 64.6% for its two GridWorld policies. Best possible: the "
        "optimal tabular behavior policy. BPG-V: the policy after 1000 iterations, mean over seeds [95% CI].\n\n"
        + md_table(["πe", "Var(g)", "best possible", "BPG-V", "share of best"], rows_policy)
        + "\n\n## 2. MSE of BPS's own estimate relative to on-policy Monte Carlo\n\n"
        "Adaptive OIS over all n trajectories collected while BPG-V learns: MSE / (Var(g)/n), mean ± SE over "
        "200 seeds. Below 1 means BPS beats on-policy Monte Carlo with the same number of trajectories.\n\n"
        + md_table(["πe"] + [f"n = {n:,}" for n in CHECKPOINTS], rows_mse)
        + "\n\n## 3. Why BPS looked no better than on-policy sampling in [2]\n\n"
        "[2]'s budget (8192 trajectories, adaptive OIS over all of them), 1000 seeds. [2]'s BPG is the repo's "
        "BehaviorPolicyGradient: plain SGD on −IS²·Σ log π with no baseline, rewards divided by v and a first "
        "batch of 9. A seed diverges when its final behavior policy has Var_IS > 2·Var(g).\n\n"
        + md_table(["πe", "BPG settings", "MSE / MSE(OS-MC) [95% CI]", "median final Var_IS / Var(g)",
                    "seeds diverged"], rows_baseline) + "\n")
    axes[0].set(xscale="log", xlabel="BPG-V iteration + 1", ylabel="Var_IS(θ) / Var(g)  (median over seeds)",
                title="Behavior policy (dotted: best possible)")
    axes[1].axhline(1, color="k", lw=1)
    axes[1].set(xscale="log", xlabel="n (trajectories)", ylabel="MSE / MSE(OS-MC)", title="BPS estimate vs OS-MC")
    axes[0].legend(fontsize=8, loc="center right")
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "bps_reproduction.png", dpi=150)
    (OUT / "bps_reproduction.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
