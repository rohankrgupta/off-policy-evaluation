"""Reproduce ROS ([2]): Monte Carlo on OS and ROS data, against the ROS repository's stored runs.

Domains and evaluation policies are the repo's: GridWorld with GridWorld_5000, and the 30-arm
MultiBandit with MultiBandit_5000. For OS (alpha = 0) and ROS (alpha = 1 ... 1e6) we run 1000 seeds
of n = 16 ... 8192 trajectories. We report the relative MSE, MSE / v^2 (the repo's metric), against
the exact DP value, next to the repo's stored runs (200 seeds).

Run:    python ros_reproduction.py     (about 15 s)
Output: results/ros_reproduction.md, results/ros_reproduction.png
"""
from pathlib import Path

import matplotlib
import numpy as np

from core import envs, exact, repo_io, sampling, stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / "results"
NS = 2 ** np.arange(4, 14)                      # n = 16 ... 8192 trajectories
SHOW = (512, 2048, 8192)
ALPHAS = (0.0, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6)
POWERS = {1.0: "1", 10.0: "10", 1e2: "10²", 1e3: "10³", 1e4: "10⁴", 1e5: "10⁵", 1e6: "10⁶"}
N_SEEDS = 1000
DOMAINS = {          # environment, repo policy, repo data folder, repo's ground truth (1e6 rollouts)
    "GridWorld": (envs.gridworld, "GridWorld_5000", "Gridworld", 2.924171),
    "MultiBandit": (envs.multibandit, "MultiBandit_5000", "bandit", 0.7849543),
}


def method(alpha):
    return "OS-MC" if alpha == 0 else f"ROS-MC α={POWERS[alpha]}"


def our_runs(mdp, logits, v):
    """Relative MSE per alpha, with 95% bootstrap CIs and the slope of log MSE vs log n."""
    out = {}
    for alpha in ALPHAS:
        sq = ((sampling.run(mdp, logits, alpha, NS, N_SEEDS, stream="e1") - v) / v) ** 2
        lo, hi = stats.bootstrap_ci(sq)
        out[method(alpha)] = {"mse": sq.mean(0), "lo": lo, "hi": hi, "slope": stats.slope(NS, sq.mean(0))}
    return out


def repo_runs(policy, folder, mean_length):
    """The repo's stored OS and ROS runs (alpha = 1 ... 1e4): relative MSE and its SE over 200 seeds.
    The repo checkpoints at environment steps; we convert them to trajectories with the exact mean length."""
    files = {"OS-MC": "OnPolicySampler1_MonteCarlo",
             **{method(10.0 ** i): f"RobustOnPolicySampler{i}_MonteCarlo" for i in range(5)}}
    out = {}
    for name, suffix in files.items():
        est, steps = repo_io.stored_run(folder, f"{policy}_{suffix}")
        sq = (est - 1.0) ** 2                     # the repo stores estimate / v
        out[name] = {"n": steps / mean_length, "mse": sq.mean(0), "se": sq.std(0, ddof=1) / np.sqrt(len(sq))}
    return out


def table(ours, repo):
    rows = []
    for name, r in ours.items():
        row = [name]
        for n in SHOW:
            i = list(NS).index(n)
            row.append(f"{r['mse'][i]:.3g} [{r['lo'][i]:.2g}, {r['hi'][i]:.2g}]")
            if name in repo:
                j = int(np.argmin(np.abs(repo[name]["n"] - n)))
                row.append(f"{repo[name]['mse'][j]:.3g} ± {repo[name]['se'][j]:.1g}")
            else:
                row.append("–")
        rows.append(row + [f"{r['slope']:+.2f}"])
    header = ["method"] + [h for n in SHOW for h in (f"ours, n = {n}", f"repo, n ≈ {n}")] + ["slope (ours)"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def plot(ax, domain, ours, repo):
    for alpha, color in zip((0.0, 1e2, 1e3, 1e4, 1e5), ("k", "tab:blue", "tab:orange", "tab:green", "tab:red")):
        name, r = method(alpha), ours[method(alpha)]
        ax.plot(NS, r["mse"], color=color, lw=1.5, label=name)
        ax.fill_between(NS, r["lo"], r["hi"], color=color, alpha=0.15, lw=0)
        if name in repo:
            keep = repo[name]["n"] >= NS[0] * 0.9
            ax.plot(repo[name]["n"][keep], repo[name]["mse"][keep], "o", color=color, ms=3.5, mfc="none")
    ax.set(xscale="log", yscale="log", title=domain, xlabel="n (trajectories)")
    ax.grid(alpha=0.3, which="both")


def main():
    OUT.mkdir(exist_ok=True)
    text = ["# ROS reproduction: relative MSE (MSE / v²), ours vs the ROS repository\n",
            f"Ours: {N_SEEDS} seeds, exact DP ground truth, 95% bootstrap CI. Repo: its stored runs (200 seeds), "
            "mean ± SE at the checkpoint closest to n. Slope: of log MSE against log n over n = 1024 … 8192 "
            "(−1 is the Monte Carlo rate).\n"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (domain, (make, policy, folder, repo_truth)) in zip(axes, DOMAINS.items()):
        mdp = make()
        logits = envs.repo_policy_logits(policy, mdp)
        pe = exact.softmax(logits)
        v, var = exact.value(mdp, pe), exact.return_variance(mdp, pe)
        ours = our_runs(mdp, logits, v)
        repo = repo_runs(policy, folder, exact.mean_length(mdp, pe))
        note = (f"Exact v = {v:.6f} and Var(g)/v² = {var / v ** 2:.4f}, so OS-MC has relative MSE "
                f"{var / v ** 2:.4f}/n ({var / v ** 2 / 8192:.3g} at n = 8192). The repo's ground truth "
                f"({repo_truth}) has squared relative error {((repo_truth - v) / v) ** 2:.1e}.")
        if domain == "MultiBandit":
            # Perfect tracking removes the variance of which arm is pulled; only the reward noise remains.
            limit = pe[0] @ mdp.reward_std[0, :, 1] ** 2 / v ** 2
            note += (f" With perfect tracking only the reward noise remains: Σ_a πe(a) σ_a² / v² = {limit:.4f}/n "
                     f"({limit / 8192:.3g} at n = 8192).")
        text.append(f"## {domain} (πe = {policy})\n\n{note}\n\n" + table(ours, repo) + "\n")
        plot(ax, domain, ours, repo)
    axes[0].set_ylabel("relative MSE (MSE / v²)")
    axes[0].legend(fontsize=8, title="lines: ours (95% CI)   circles: repo", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "ros_reproduction.png", dpi=150)
    (OUT / "ros_reproduction.md").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
