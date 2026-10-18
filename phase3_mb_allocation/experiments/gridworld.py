"""Model-based estimation with an optimized allocation, against ROS-MC and ReVar, at equal environment steps.

Domains: GridWorld with slip 0.1, with slip 0.2, and with slip 0.1 plus Gaussian reward noise
(std 11.5) on every transition into the trap. pi_e = the repo's epsilon-greedy 0.7 policy (pi_e-B).

Every method gets the same budget of T environment steps, with T = 8192 on-policy episodes' worth.
Results are reported at T/4, T/2 and T.

Methods (estimator in brackets):
  ROS-MC [MC]              ROS tracking pi_e: the best method of Phases 1-2.
  OS-MB [MB]               on-policy data: the estimator alone.
  ReVar [MB]               ReVar's allocation rule, i.i.d. sampling.
  ours [MB]                the per-step MB-optimal allocation, i.i.d. sampling.
  ours + ROS [MB]          the same, tracked with ROS.
Each allocation is run twice:
  * oracle (*): computed from the true MDP;
  * learned (^): computed from a model fitted to T/8 on-policy stage-1 steps per seed, mixed 10% with
    pi_e. The stage-1 steps count toward the budget and are pooled into the MB estimate.
alpha (ROS) is tuned per method on 500 separate seeds; results use 1000 seeds.

Run:    python -m phase3_mb_allocation.experiments.gridworld     (about 6 min)
Output: results/gridworld.md, results/gridworld.png
"""
import multiprocessing as mp
from pathlib import Path

import matplotlib
import numpy as np

from phase1_reproduction.core import envs, exact, stats
from phase3_mb_allocation.core import allocation, collect

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
ALPHAS = (1e2, 1e3, 1e4, 1e5)
N_EVAL, N_TUNE = 1000, 500


def domains():
    pi_b = envs.eps_greedy_logits(envs.repo_policy_logits("GridWorld_5000", envs.gridworld()), 0.7,
                                  envs.gridworld().terminal)
    trap = envs.gridworld(slip=0.1)
    trap.reward_std[:, :, envs.TRAP] = np.where(trap.P[:, :, envs.TRAP] > 0, 11.5, 0.0)
    return [("slip 0.1", envs.gridworld(slip=0.1), pi_b), ("slip 0.2", envs.gridworld(slip=0.2), pi_b),
            ("slip 0.1 + noisy trap", trap, pi_b)]


def _fit(job):
    mdp, pi_e, seed_stats, rule = job
    return allocation.learned_target(mdp, pi_e, seed_stats, rule)


def learned(pool, mdp, pi_e, stage1, rule):
    n = len(stage1["starts"])
    jobs = [(mdp, pi_e, {key: val[i] for key, val in stage1.items()}, rule) for i in range(n)]
    return np.stack(pool.map(_fit, jobs, chunksize=10))


def run_domain(pool, name, mdp, th_e):
    pe = exact.softmax(th_e)
    v, var = exact.value(mdp, pe), exact.return_variance(mdp, pe)
    _, length = allocation.mb_variance(mdp, pe, pe)
    T = int(8192 * length)
    cks, t1 = np.array([T // 4, T // 2, T]), T // 8
    scale = cks / (length * var)                 # MSE * scale = MSE relative to on-policy MC

    stage1 = {seeds: collect.run(mdp, th_e, th_e, 0.0, [t1], n, f"p3-stage1-{seeds}")["stats"]
              for seeds, n in (("eval", N_EVAL), ("tune", N_TUNE))}
    targets = {"ours*": (allocation.optimal_target(mdp, pe), False),
               "ReVar*": (np.log(np.clip(allocation.revar_target(mdp, pe), 1e-300, None)), True)}
    for rule, label, timed in (("ours", "ours^", False), ("revar", "ReVar^", True)):
        targets[label] = ({s: learned(pool, mdp, pe, stage1[s], rule) for s in ("eval", "tune")}, timed)

    def estimates(target, alpha, seeds, estimator="MB"):
        """Estimates (seeds, checkpoints) from data collected with `target` ("πe" or a key of targets)."""
        tgt, timed = targets[target] if target in targets else (th_e, False)
        if isinstance(tgt, dict):                # learned: one target per seed; stage 2 gets the rest of the budget
            out = collect.run(mdp, th_e, tgt[seeds], alpha, cks - t1, N_EVAL if seeds == "eval" else N_TUNE,
                              f"p3-{seeds}", timed, init=stage1[seeds])
        else:
            out = collect.run(mdp, th_e, tgt, alpha, cks, N_EVAL if seeds == "eval" else N_TUNE, f"p3-{seeds}", timed)
        return out[estimator]

    def tuned_alpha(target, estimator="MB"):         # ROS strength with the lowest MSE on the tuning seeds
        return min(ALPHAS, key=lambda a: ((estimates(target, a, "tune", estimator)[:, -1] - v) ** 2).mean())

    methods = {}   # label -> (allocation, sampling, alpha, estimates (seeds, checkpoints))
    a = tuned_alpha("πe", "MC")
    methods["ROS-MC"] = ("πe", "ROS", a, estimates("πe", a, "eval", "MC"))
    methods["OS-MB"] = ("πe", "i.i.d.", 0.0, estimates("πe", 0.0, "eval"))
    for kind in ("*", "^"):
        methods[f"ReVar{kind}"] = (f"ReVar{kind}", "i.i.d.", 0.0, estimates(f"ReVar{kind}", 0.0, "eval"))
        methods[f"ours{kind}"] = (f"ours{kind}", "i.i.d.", 0.0, estimates(f"ours{kind}", 0.0, "eval"))
        a = tuned_alpha(f"ours{kind}")
        methods[f"ours{kind} + ROS"] = (f"ours{kind}", "ROS", a, estimates(f"ours{kind}", a, "eval"))
    def predicted(b):    # exact first-order MSE of MB relative to on-policy MC at equal steps
        F, L = allocation.mb_variance(mdp, pe, b)
        return F * L / (length * var)

    pred = {"OS-MB": predicted(pe), "ReVar*": predicted(allocation.revar_target(mdp, pe)),
            "ours*": predicted(exact.softmax(targets["ours*"][0]))}
    print(f"{name} done", flush=True)
    return {"name": name, "v": v, "var": var, "scale": scale, "methods": methods, "predicted": pred,
            "ceiling": allocation.per_step_ceiling(mdp, pe)}


def md(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows])


def main():
    OUT.mkdir(exist_ok=True)
    with mp.get_context("spawn").Pool(8) as pool:
        results = [run_domain(pool, *d) for d in domains()]
    sections, contrast_rows = [], []
    pairs = [("ours*", "ReVar*"), ("ours^", "ReVar^"), ("ours* + ROS", "ours*"), ("ours^ + ROS", "ours^"),
             ("OS-MB", "ROS-MC"), ("ours^", "ROS-MC"), ("ours^ + ROS", "ROS-MC")]
    for D in results:
        rows = []
        for label, (_, sampling_, alpha, est) in D["methods"].items():
            rel = ((est - D["v"]) ** 2).mean(0) * D["scale"]
            pred = D["predicted"].get(label.replace(" + ROS", ""), None) if label != "ROS-MC" else None
            rows.append([label, sampling_, "–" if alpha == 0 else f"{alpha:g}", "–" if pred is None else f"{pred:.3f}"]
                        + [f"{x:.3f}" for x in rel])
        p = D["predicted"]
        sections.append(f"### {D['name']}\n\nPer-step gain over on-policy MB: at most {D['ceiling']:.2f}× for any behavior "
                        f"(1 + CV²); the oracle allocation achieves {p['OS-MB'] / p['ours*']:.2f}× (ReVar's: "
                        f"{p['OS-MB'] / p['ReVar*']:.2f}×).\n\n"
                        + md(["method", "sampling", "α", "predicted", "T/4", "T/2", "T"], rows))
        for x, y in pairs:
            r, lo, hi = stats.paired_mse_ratio(D["methods"][x][3][:, -1], D["methods"][y][3][:, -1], D["v"])
            contrast_rows.append([D["name"], f"{x} vs {y}", f"{r:.3f} [{lo:.3f}, {hi:.3f}] "
                                  + ("✓" if hi < 1 else "✗" if lo > 1 else "~")])
    text = ("# Model-based estimation with an optimized allocation (GridWorld, equal environment steps)\n\n"
            "## Paired MSE ratios at the full budget T [95% CI]\n\n"
            "✓ the first method is better, ✗ worse, ~ inconclusive. * = oracle allocation, ^ = learned from stage-1 "
            "data (counted in the budget).\n\n" + md(["domain", "comparison", "MSE ratio"], contrast_rows)
            + "\n\n## MSE relative to on-policy Monte Carlo at the same number of environment steps\n\n"
            "On-policy MC = 1; lower is better. 1000 seeds. Predicted: exact first-order value for oracle "
            "allocations.\n\n" + "\n\n".join(sections) + "\n")
    (OUT / "gridworld.md").write_text(text)
    print(text)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=False)
    for ax, D in zip(axes, results):
        labels = list(D["methods"])
        rel = [((D["methods"][m][3][:, -1] - D["v"]) ** 2) * D["scale"][-1] for m in labels]
        lo_hi = np.array([stats.bootstrap_ci(x) for x in rel])
        mean = np.array([x.mean() for x in rel])
        colors = ["k", "0.55"] + ["tab:blue", "tab:orange", "tab:red"] * 2
        ax.bar(range(len(labels)), mean, color=colors, alpha=0.85,
               yerr=[mean - lo_hi[:, 0], lo_hi[:, 1] - mean], capsize=3)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_title(D["name"])
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("MSE / MSE(on-policy MC), equal steps")
    fig.suptitle("Equal environment steps. * oracle allocation, ^ learned (stage 1 counted). 95% CI.", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "gridworld.png", dpi=150)


if __name__ == "__main__":
    main()
