# Phase 2.5: Coordinated BPS — behavior policy search with quasi-Monte Carlo sampling

Behavior policy search (BPS, [1]) lowers the variance of a policy's value estimate by collecting data with a different behavior policy and reweighting it. Phase 2 showed that tracking BPS's target with ROS fails. Here we coordinate BPS's samples instead: all episodes of the behavior policy run in lockstep, and at every step the episodes that are in similar states receive evenly spread action draws (Array-RQMC; randomized QMC for policy evaluation is due to Arnold et al., 2022). Each trajectory is still distributed exactly as the behavior policy, so importance sampling stays unbiased and independent repeats give valid error bars.

**The method.**
1. Learn the behavior policy q once: by BPG as in [1], or from a pilot of on-policy episodes by the offline fit below (about 100× fewer episodes). The same q is reused across evaluations and conditions.
2. Run n episodes of q in lockstep. At each step, sort the running episodes by state (or by a value estimate when states are continuous) and give the episode of rank i the action uniform frac(φ(i) + shift_t), with φ the van der Corput sequence and a fresh random shift per step.
3. Estimate v(π_e) by ordinary importance sampling (OIS) or per-decision IS (PDIS).
4. When states repeat (discrete states): merge past weights, i.e. replace each episode's importance weight by the average over the episodes in the same state. This is marginalized importance sampling (Xie et al., 2019) computed online; it is exact there and adds up to 1.7×. In continuous states it is biased and is not used.

**Samplers** (`core/`): per-state QMC (a shifted van der Corput sequence per state); trajectory-level Sobol RQMC, Latin hypercube and antithetic pairs; Array-RQMC with weight-aware sorting, prefix-weight merging and Q-ordered actions; the CartPole samplers (i.i.d., ROS, Array-RQMC, in-budget BPS).

## Results

### 1. Discrete states (GridWorld): BPS works once its samples are coordinated

n·MSE/Var(g) at n = 8192, 1000 seeds (on-policy Monte Carlo = 1):

| Method | det πe-A | det πe-B | slip 0.1 | slip 0.2 | noisy trap |
|---|---|---|---|---|---|
| BPS + OIS, i.i.d. ([1]) | 0.90 | 0.29 | 0.39 | 0.57 | 0.54 |
| BPS + ROS (Phase 2) | 0.030 | 0.27 | 0.39 | 0.55 | 0.53 |
| **BPS + Array-RQMC, merged, PDIS** | **0.013** | **0.0049** | **0.204** | **0.335** | **0.364** |
| … with Q-ordered actions | 0.009 | 0.0022 | 0.191 | 0.357 | 0.359 |
| ROS-MC (on-policy) | 0.0012 | 0.0001 | 0.218 | 0.379 | 0.471 |
| *Model-based estimate, same data (exact)* | *0* | *0* | *0.163* | *0.326* | *0.292* |

Paired MSE ratios of the merged version at n = 8192, with 95% CIs (✓ means better):

| vs | det πe-A | det πe-B | slip 0.1 | slip 0.2 | noisy trap |
|---|---|---|---|---|---|
| BPS + OIS | 0.014 ✓ | 0.017 ✓ | 0.53 ✓ | 0.59 ✓ | 0.68 ✓ |
| BPS + ROS | 0.43 ✓ | 0.019 ✓ | 0.53 ✓ | 0.61 ✓ | 0.69 ✓ |
| ROS-MC | 10.4 ✗ | 45 ✗ | 0.94 ~ | 0.89 ✓ | 0.77 ✓ |

- **Against the original BPS methods:** 1.5–70× better than BPS + OIS on every domain, 1.5–54× better than BPS + ROS at n = 8192 (on πe-A, BPS + ROS is still better at small n). Unbiased (|bias|/SE ≤ 1.4).
- **Against ROS-MC:** matches or beats it per episode in the noisy domains (significant on slip 0.2 and the noisy trap). In deterministic domains ROS-MC wins by 7–20× even with Q-ordering: its rate there is n⁻², ours n⁻¹·⁸.
- **Why it works.** Coordination removes the variance of which action is drawn in each state. PDIS stops later weights from multiplying earlier rewards. Merging removes the spread of weights among episodes that reach the same state; with merging the estimate gets close to the model-based estimate on the same data (0.335 vs 0.326 on slip 0.2).
- **A target built for coordination doesn't help.** The variance left after coordination (J, `core/j_target.py`) is the variance of the doubly robust estimator with the true Q (Jiang and Li, 2016). Minimizing it gives a target within 3.5–6% of BPS's, and the measured difference is not significant (`results/j_target.md`); J predicts the measured variance within 0–10%. So coordinated PDIS behaves like DR with the true Q, without learning Q.

### 2. Continuous states (CartPole): coordinated BPS beats ROS

The ROS repository's two CartPole benchmarks with its trained policies (`core/cartpole.py`): CartPole-v0 with a softmax policy over two pushes, and its continuous-action version with a Gaussian policy. Episodes are sorted by a linear value estimate; weights are not merged. Force noise of 3 N makes the dynamics stochastic. n·MSE/Var(g) at n = 4096, 1000 seeds, with the BPS target given (learned beforehand by BPG; `results/continuous/cartpole.md`, `cartpole_continuous.md`):

| Method | discrete, 0 N | discrete, 3 N | continuous actions, 0 N | continuous actions, 3 N |
|---|---|---|---|---|
| OS-MC | 1.00 | 1.07 | 1.02 | 1.05 |
| ROS-MC (α tuned at each n) | 0.265 | 0.341 | 0.638 | 0.674 |
| BPS + OIS, i.i.d. ([1]) | 0.251 | 0.328 | 0.409 | 0.420 |
| Array-RQMC, on-policy (MC) | 0.741 | 0.787 | 0.915 | 0.809 |
| **BPS + Array-RQMC, OIS** | **0.158** | **0.242** | **0.341** | **0.392** |
| paired ratio vs ROS-MC (95% CI) | 0.60 ✓ | 0.71 ✓ | 0.54 ✓ | 0.58 ✓ |
| paired ratio vs BPS + OIS | 0.63 ✓ | 0.74 ✓ | 0.83 ✓ | 0.93 ~ |

- **1.4–1.9× better than ROS per episode, 1.16–1.5× at equal environment steps** (the BPS target lengthens episodes by about 20%). Unbiased (|bias|/SE ≤ 1.2), coverage 93–97%.
- **Neither ingredient alone does it:** on-policy Array-RQMC gains little (0.74–0.92), and BPS alone is at best level with ROS.
- **OIS beats PDIS here** (by 1.3–1.9×), the opposite of GridWorld.
- **Merging over neighbours in value order is biased** (|bias|/SE of 20–44, coverage 3–46%), so merging stays an add-on for discrete states.
- **Cost:** coordination adds about 6% wall-clock per episode over i.i.d. BPS sampling (12.7 vs 12.0 µs); ROS costs 1.6× more per episode (20 µs) for its per-step gradient.
- **Learning the target online inside the budget does not pay** at these n: the repository's BPG on the first half of the episodes, then Array-RQMC, gives 0.43–0.46 at n = 4096 against ROS's 0.27–0.34, and on the continuous task one run blew up (9.6).

### 3. The target is learned once and reused

**Offline fit** (`core/offline_bps.py`). BPG uses each episode for one gradient step; a good CartPole target took it 3.2M episodes. Instead we collect one on-policy pilot and fit q to it: minimize the pilot's importance-sampling estimate of q's variance, (1/n)Σ g_i² π_e(τ_i)/q(τ_i), plus a KL pull toward π_e, by L-BFGS over the policy's output layer only. [1] tried reusing earlier batches this way and found it unstable; the restriction to the output layer and the KL term are what make it work. Quality = Var_q(OIS)/Var(g), on 10⁶ fresh episodes (`results/continuous/offline_target.md`):

| Pilot episodes | 256 | 512 | 1024 | 2048 | BPG, 3.2M episodes |
|---|---|---|---|---|---|
| Quality of the fitted target | 1.15 | 0.49 | 0.39 | 0.27 | 0.26 |

Fits take 2–17 s. Fitting the whole network overfits at every pilot size, and below 512 episodes the fit is no better than π_e. In the exact GridWorld check, fits from small pilots without the KL term have true variances of 10³–10⁶⁰× on-policy that the empirical MSE does not show, so the shrinkage is not optional.

**Transfer.** A target learned under one level of dynamics noise is as good as one learned natively under another:

| Target | used at 0 N | used at 3 N | used at 6 N |
|---|---|---|---|
| BPG, learned at 0 N | 0.257 | 0.329 | 0.497 |
| BPG, learned at 3 N | 0.265 | 0.331 | 0.487 |
| offline fit, 2048 pilot episodes at 0 N | 0.268 | 0.337 | 0.499 |
| offline fit, 2048 pilot episodes at 3 N | 0.288 | 0.352 | 0.502 |

Quality falls with noise only because more environment noise leaves less for any behavior policy to remove.

**Break-even.** With the target given, coordinated BPS reaches ROS's MSE on CartPole with 60% of ROS's episodes (73% of its environment steps). A 2048-episode pilot is therefore repaid after about 1.3 evaluations of 4096 episodes (about 2 at equal steps); every further evaluation of the same policy, under the same or a perturbed environment, gains the full 1.4–1.9×. The n at which the pipeline beats ROS with the pilot counted in a single evaluation is being measured (`results/continuous/cartpole_breakeven.md`, pending).

### 4. Smooth continuous actions: QMC beats ROS and PROPS

Linear-quadratic control with a Gaussian policy (`core/lq.py`), where the value is exact. With continuous actions, the action is a smooth function of its uniform, which is where QMC is strongest.

Paired MSE ratios at n = 4096, 1000 seeds (✓ means QMC is better):

| Configuration | Sobol vs ROS | Sobol vs PROPS | Array-RQMC vs ROS |
|---|---|---|---|
| 1-D, H = 10, no environment noise | **0.0054 ✓** | **0.0015 ✓** | **0.011 ✓** |
| 1-D, H = 10, environment noise σ_w = 0.1 | **0.57 ✓** | **0.26 ✓** | **0.56 ✓** |
| 1-D, H = 10, σ_w = 0.3 | 0.94 ~ | **0.82 ✓** | 0.99 ~ |
| 1-D, H = 50, σ_w = 0.1 | 0.93 ~ | **0.21 ✓** | 0.89 ~ |
| 4-D, H = 50, σ_w = 0.1 | 0.96 ~ | **0.94 ✓** | 0.95 ~ |

- When the agent's own noise dominates, Sobol RQMC converges at n⁻². It is up to 185× better than ROS and 670× better than PROPS.
- The advantage shrinks as environment noise grows. In the 4-D, H = 50 setting no method gets far below i.i.d. sampling.
- QMC's gain over i.i.d. sampling on LQR is Arnold et al.'s result; the comparison with ROS and PROPS is the part added here.

### 5. Unbiased estimates with valid error bars

Data collected with ROS or PROPS give biased Monte Carlo estimates, because each action depends on all earlier data. When their variance is small, that bias makes confidence intervals undercover. The RQMC designs keep every trajectory exactly distributed as the sampling policy.

| | bias/SE | coverage of 95% intervals from 10 independent runs |
|---|---|---|
| ROS, deterministic GridWorld | −12 to −20 | 55–78% |
| ROS / PROPS, LQ (1-D, H = 10, σ_w ≤ 0.1) | −5 to −14 | 76–92% |
| Sobol RQMC and Array-RQMC (all domains) | within ±2 except one cell (+3.1) | 88–99% |

### Where each method wins
- **Deterministic discrete domains, on-policy:** ROS. Its adaptive per-state balancing reaches n⁻² there.
- **Noisy or continuous-state domains, when the behavior policy can be learned once:** coordinated BPS.
- **Smooth continuous actions:** QMC, even on-policy.
- **Small budgets with no pilot and no logged data:** ROS; learning the target inside a few thousand episodes does not pay.

**Tried, neutral or negative:**
- Per-state QMC is a weaker, non-adaptive ROS: with ROS's α tuned at each n, it wins only on πe-A at n ≤ 256. With BPS's weights it blows up.
- Latin hypercube and antithetic sampling are weaker than Sobol. Antithetic is 2× worse than i.i.d. in LQ, whose returns are symmetric in the action noise.
- Adding RQMC on top of ROS gives nothing.
- On-policy in the discrete GridWorld, ROS beats every QMC sampler.
- Per environment step in GridWorld, coordinated BPS loses to ROS-MC, because BPS's target lengthens episodes (25–27 steps vs 19–21).
- Q-ordering used the exact Q (a best case); in practice Q would have to be estimated.
- Weight merging over nearest neighbours in continuous states is biased, and the bias does not shrink with n.
- The cross-entropy target (|return|-weighted behavior cloning of π_e's own episodes) equals the BPS target within 1–2% in the exact GridWorld check, as [1] found for BPG-KL, but from small pilots it is worse than the offline fit.

## Run

From the repository root:

```bash
python -m phase2_5_qmc.experiments.gridworld                   # 6 min: QMC samplers vs ROS on the GridWorld (+ figure)
python -m phase2_5_qmc.experiments.array_rqmc                  # 4 min: Array-RQMC, bias and interval coverage
python -m phase2_5_qmc.experiments.j_target                    # 8 min: a target designed for coordination
python -m phase2_5_qmc.experiments.merging                     # 8 min: prefix-weight merging and Q-ordered actions
python -m phase2_5_qmc.experiments.continuous.lq               # 1 hour: continuous actions, QMC vs ROS vs PROPS
python -m phase2_5_qmc.experiments.continuous.cartpole discrete    # 45 min: CartPole, discrete actions
python -m phase2_5_qmc.experiments.continuous.cartpole continuous  # 45 min: CartPole, continuous actions
python -m phase2_5_qmc.experiments.continuous.offline_target   # 6 min: the offline target fit and its transfer
```

## Code

```
core/qmc.py            per-state QMC and trajectory-level designs (Sobol, Latin hypercube, antithetic)
core/array_rqmc.py     Array-RQMC in lockstep, with weight-aware sorting, prefix-weight merging, Q-ordering (tabular)
core/j_target.py       the variance left after coordination (J), its exact gradient and its minimizer
core/lq.py             linear-quadratic testbed (exact value) and its samplers, including ROS and PROPS
core/cartpole.py       the ROS repo's CartPole tasks and policies: i.i.d., ROS, Array-RQMC and in-budget BPS samplers
core/offline_bps.py    the offline fit of the BPS target from an on-policy pilot
```

Environments, exact DP, ROS toward πe and BPS's target come from Phases 1 and 2.

**References.**
- S. M. R. Arnold, P. L'Ecuyer, L. Chen, Y. Chen, F. Sha. *Policy Learning and Evaluation with Randomized Quasi-Monte Carlo.* AISTATS 2022.
- P. L'Ecuyer, C. Lécot, B. Tuffin. *A Randomized Quasi-Monte Carlo Simulation Method for Markov Chains.* Operations Research 2008.
- D. Precup, R. S. Sutton, S. Singh. *Eligibility Traces for Off-Policy Policy Evaluation.* ICML 2000 (per-decision IS).
- T. Xie, Y. Ma, Y.-X. Wang. *Towards Optimal Off-Policy Evaluation for Reinforcement Learning with Marginalized Importance Sampling.* NeurIPS 2019.
- N. Jiang, L. Li. *Doubly Robust Off-Policy Value Evaluation for Reinforcement Learning.* ICML 2016.
- N. E. Corrado, J. P. Hanna. *On-Policy Policy Gradient Reinforcement Learning Without On-Policy Sampling* (PROPS).
