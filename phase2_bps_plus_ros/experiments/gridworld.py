"""BPS + ROS in a multi-step problem: the ROS repository's GridWorld.

Domains: the deterministic GridWorld with pi_e-A (GridWorld_5000) and pi_e-B (epsilon-greedy 0.7), and
the slippery GridWorld (p = 0.1) with pi_e-B.

Methods (1000 seeds, n up to 8192 trajectories, alpha tuned per method on 500 other seeds):
  OS-MC, ROS-MC        on-policy Monte Carlo, i.i.d. or with ROS tracking pi_e (Phase 1 code);
  BPS-OIS, BPS-R       i.i.d. trajectories from the learned BPS target, weighted by OIS (B) or regression IS (R);
  BPS+ROS-A/B/R        ROS tracking the learned BPS target, with the three weightings.
The learned target is BPG-V on 1024 trajectories per seed. The exact BPS target (BPS*) is also run.

Run:    python -m phase2_bps_plus_ros.experiments.gridworld     (about 5 min)
Output: results/gridworld.md, results/gridworld.png
"""
from pathlib import Path

import matplotlib
import numpy as np

from phase1_reproduction.core import envs, exact, sampling, stats
from phase2_bps_plus_ros.core import bps_ros

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
NS = 2 ** np.arange(4, 14)                      # n = 16 ... 8192
SHOW = (2, 5, 8, 9)                             # table columns: n = 64, 512, 4096, 8192
ALPHAS = (0.0, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6)
N_EVAL, N_TUNE = 1000, 500


def domains():
    g = envs.gridworld()
    pi_a = envs.repo_policy_logits("GridWorld_5000", g)
    pi_b = envs.eps_greedy_logits(pi_a, 0.7, g.terminal)
    # name, environment, pi_e, BPG-V batch size for the learned target (Phase 1: batch 100 needed for pi_e-A)
    return [("πe-A", g, pi_a, 100), ("πe-B", g, pi_b, 10), ("πe-B, slippery", envs.gridworld(slip=0.1), pi_b, 10)]


def run_domain(name, mdp, th_e, k):
    pe = exact.softmax(th_e)
    v, var = exact.value(mdp, pe), exact.return_variance(mdp, pe)
    oracle = bps_ros.oracle_bps_target(mdp, th_e)
    targets = {"BPS": (bps_ros.learned_bps_target(mdp, th_e, N_EVAL, f"stage1-e4-{name}", k=k),
                       bps_ros.learned_bps_target(mdp, th_e, N_TUNE, f"stage1-e4-tune-{name}", k=k)),
               "BPS*": (oracle, oracle)}
    runs = {}   # (method, alpha) -> (eval estimates (seeds, n), tuning MSE at n = 8192)
    for alpha in ALPHAS:
        ev = sampling.run(mdp, th_e, alpha, NS, N_EVAL, "e4")
        tu = sampling.run(mdp, th_e, alpha, NS, N_TUNE, "tune-e4")
        runs[("MC", alpha)] = (ev, ((tu[:, -1] - v) ** 2).mean())
        for target, (t_eval, t_tune) in targets.items():
            ev = bps_ros.run(mdp, th_e, t_eval, alpha, NS, N_EVAL, "e4")
            tu = bps_ros.run(mdp, th_e, t_tune, alpha, NS, N_TUNE, "tune-e4")
            for est in ("A", "B", "R"):
                runs[(f"{target}-{est}", alpha)] = (ev[est], ((tu[est][:, -1] - v) ** 2).mean())
        print(f"  {name}: α = {alpha:g} done", flush=True)

    def tuned(method, alphas=ALPHAS[1:]):
        return min(alphas, key=lambda a: runs[(method, a)][1])

    rows = {"OS-MC": ("MC", 0.0), "ROS-MC": ("MC", tuned("MC"))}
    for t in targets:
        rows.update({f"{t}-OIS": (f"{t}-B", 0.0), f"{t}-R": (f"{t}-R", 0.0),
                     f"{t}+ROS-A": (f"{t}-A", tuned(f"{t}-A", (1.0, 10.0, 1e2))),
                     f"{t}+ROS-B": (f"{t}-B", tuned(f"{t}-B")), f"{t}+ROS-R": (f"{t}-R", tuned(f"{t}-R"))})
    best_iid = exact.is_variance(mdp, pe, exact.softmax(oracle)) / var
    return {"name": name, "v": v, "var": var, "runs": runs, "rows": rows, "tuned": tuned, "best_iid": best_iid}


def nmse(D, label):
    method, alpha = D["rows"][label]
    return NS * ((D["runs"][(method, alpha)][0] - D["v"]) ** 2).mean(0) / D["var"]


def md(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows])


def main():
    OUT.mkdir(exist_ok=True)
    results = [run_domain(*d) for d in domains()]
    sections, contrasts = [], []
    for D in results:
        rows = []
        for label, (method, alpha) in D["rows"].items():
            y = nmse(D, label)
            rows.append([label, f"{alpha:g}"] + [f"{y[i]:.3g}" for i in SHOW] + [f"{stats.slope(NS, y / NS):+.2f}"])
        sections.append(f"### {D['name']}\n\nv = {D['v']:.3f}, Var(g) = {D['var']:.1f}. Best possible i.i.d. BPS "
                        f"(exact target, OIS): {D['best_iid']:.3f}.\n\n"
                        + md(["method", "α", "n = 64", "n = 512", "n = 4096", "n = 8192", "slope"], rows))
        for t in ("BPS", "BPS*"):
            est = min(("B", "R"), key=lambda e: D["runs"][(f"{t}-{e}", D["tuned"](f"{t}-{e}"))][1])
            x = D["runs"][(f"{t}-{est}", D["tuned"](f"{t}-{est}"))][0][:, -1]
            row = [D["name"], f"{t}+ROS-{est}"]
            for comp in (("MC", D["tuned"]("MC")), (f"{t}-B", 0.0), (f"{t}-R", 0.0)):
                ratio, lo, hi = stats.paired_mse_ratio(x, D["runs"][comp][0][:, -1], D["v"])
                row.append(f"{ratio:.3g} [{lo:.3g}, {hi:.3g}] " + ("✓" if hi < 1 else "✗" if lo > 1 else "~"))
            contrasts.append(row)
    text = ("# BPS + ROS on the GridWorld\n\n"
            "## Paired MSE ratios of BPS + ROS at n = 8192 [95% CI]\n\n"
            "✓ BPS + ROS is better, ✗ worse, ~ inconclusive. The tracked method uses B or R, whichever tuned better. "
            "BPS = learned target (BPG-V on 1024 trajectories per seed); BPS* = exact target.\n\n"
            + md(["domain", "method", "vs ROS-MC", "vs BPS-OIS", "vs BPS-R (i.i.d. + R)"], contrasts)
            + "\n\n## n·MSE/Var(g) by method\n\nOn-policy Monte Carlo = 1; lower is better. 1000 seeds; α tuned on "
            "500 other seeds. Slope: of log MSE against log n over n = 1024 … 8192 (−1 is the Monte Carlo rate).\n\n"
            + "\n\n".join(sections) + "\n")
    (OUT / "gridworld.md").write_text(text)
    print(text)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, D in zip(axes, results):
        for label, color in (("OS-MC", "0.5"), ("ROS-MC", "k"), ("BPS-OIS", "tab:blue"), ("BPS+ROS-B", "tab:orange"),
                             ("BPS+ROS-R", "tab:red")):
            ax.plot(NS, nmse(D, label), "o-", ms=3, color=color, label=label)
        ax.axhline(D["best_iid"], color="tab:blue", ls=":", lw=1)
        ax.set(xscale="log", yscale="log", title=D["name"], xlabel="n (trajectories)")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("n·MSE / Var(g)")
    axes[0].legend(fontsize=8)
    fig.suptitle("GridWorld: ROS-MC beats BPS + ROS (dotted: best possible i.i.d. BPS)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "gridworld.png", dpi=150)


if __name__ == "__main__":
    main()
