"""Learning the BPS target cheaply, and reusing it: CartPole, discrete actions (core/offline_bps.py).

1. Quality of the offline fit by pilot size: q is fitted (output layer, KL weight 10) to n1 on-policy pilot episodes;
   its quality Var_q(OIS)/Var(g) is measured on 1e6 fresh episodes (1.0 = on-policy; 2 pilots per size).
   Reference: the BPG target of the main CartPole experiment, learned from 3.2e6 episodes.
2. Transfer across conditions: targets learned under force noise 0 or 3 N (by BPG, and by the offline fit from 2048
   pilot episodes) are evaluated under force noise 0, 3 and 6 N, against the target learned natively at each level.

Run:    python -m phase2_5_qmc.experiments.continuous.offline_target     (about 6 min)
Output: results/continuous/offline_target.md
"""
import time
from pathlib import Path

import numpy as np

from phase2_5_qmc.core import cartpole as cp
from phase2_5_qmc.core import offline_bps as ob
from phase2_5_qmc.experiments.gridworld import md

OUT = Path(__file__).resolve().parents[2] / "results" / "continuous"
N1S = (256, 512, 1024, 2048)
NOISES = (0.0, 3.0, 6.0)
LAM = 10.0


def truth(pi_e, noise):
    r = cp.iid(pi_e, None, 100_000, 10, "offline-truth", noise)
    v = r["MC"].mean()
    return v, r["G2"].mean() - v * v


def quality(pi_e, theta, noise, v, var):
    return (cp.iid(pi_e, theta, 100_000, 10, "offline-check", noise)["OIS2"].mean() - v * v) / var


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pi_e = cp.load_pi_e()
    tv = {nz: truth(pi_e, nz) for nz in NOISES}

    rows = []                                   # 1. quality by pilot size (noise 0)
    v, var = tv[0.0]
    for n1 in N1S:
        q, secs = [], 0.0
        for seed in range(2):
            pilot = ob.collect(pi_e, n1, seed)
            t0 = time.perf_counter()
            theta = ob.learn(pi_e, pilot, LAM, "last")[0]
            secs += time.perf_counter() - t0
            q.append(quality(pi_e, theta, 0.0, v, var))
        rows.append([f"offline fit, {n1} pilot episodes", f"{np.mean(q):.3f}", f"{np.max(q):.3f}", f"{secs / 2:.0f} s"])
    t0 = time.perf_counter()
    bpg = {nz: cp.bpg_target(pi_e, nz) for nz in (0.0, 3.0)}
    secs = (time.perf_counter() - t0) / 2
    rows.append(["BPG target, 3.2e6 episodes (main experiment)", f"{quality(pi_e, bpg[0.0], 0.0, v, var):.3f}", "–",
                 f"{secs:.0f} s"])

    targets = {f"BPG, learned at {nz:g} N": bpg[nz] for nz in (0.0, 3.0)}    # 2. transfer
    for nz in (0.0, 3.0):
        targets[f"offline fit, 2048 pilot episodes at {nz:g} N"] = ob.learn(pi_e, ob.collect(pi_e, 2048, 0, noise=nz),
                                                                           LAM, "last")[0]
    transfer = [[name, *(f"{quality(pi_e, th, nz, *tv[nz]):.3f}" for nz in NOISES)] for name, th in targets.items()]
    native = ["BPG, learned natively at each level", f"{quality(pi_e, bpg[0.0], 0.0, *tv[0.0]):.3f}",
              f"{quality(pi_e, bpg[3.0], 3.0, *tv[3.0]):.3f}", "–"]

    text = ("# Learning the BPS target cheaply, and reusing it (CartPole, discrete actions)\n\n"
            "Quality = Var_q(OIS)/Var(g) on 1e6 fresh episodes; 1.0 = on-policy, lower is better.\n\n"
            "## Offline fit by pilot size (force noise 0)\n\nThe output layer of q is fitted to the pilot's importance-"
            f"sampling estimate of the variance with a KL pull toward pi_e (weight {LAM:g}); 2 pilots per size.\n\n"
            + md(["target", "quality (mean of 2)", "worst", "fit time"], rows)
            + "\n\n## Transfer across force-noise levels\n\n"
            + ", ".join(f"{nz:g} N: v = {tv[nz][0]:.3f}, Var(g) = {tv[nz][1]:.1f}" for nz in NOISES) + ".\n\n"
            + md(["target", *(f"used at {nz:g} N" for nz in NOISES)], transfer + [native]) + "\n")
    (OUT / "offline_target.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
