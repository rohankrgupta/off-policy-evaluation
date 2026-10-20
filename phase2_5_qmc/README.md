# Phase 2.5: Quasi-Monte Carlo sampling with ROS and BPS

ROS reduces the luck in which actions get sampled by steering each action toward the policy. Quasi-Monte Carlo (QMC) attacks the same luck differently: it replaces the random numbers that pick actions with evenly spread ones. Only the agent's randomness is changed; the environment's is left alone.

We compare QMC-type action samplers with ROS and PROPS, and combine them with BPS. Randomized QMC and Array-RQMC for RL policy evaluation come from Arnold et al. (2022); here we apply them in the ROS/BPS setting.

**Samplers** (`core/`):
- **Per-state QMC:** a randomly shifted van der Corput sequence per state.
- **Trajectory-level designs:** Sobol RQMC, Latin hypercube, antithetic pairs.
- **Array-RQMC:** all episodes run in lockstep. At each step, episodes are sorted by state and get stratified action uniforms, so episodes in the same state receive evenly spread actions. With importance weights it has three options:
  - **weight-aware sorting:** within a state, episodes are ordered by current weight;
  - **prefix-weight merging:** each episode's past weight is replaced by the average over its state group. This is still unbiased, because futures from the same state are exchangeable;
  - **Q-ordered actions:** actions are laid out along [0, 1) by π_e/q·Q.

## Results

### 1. BPS works once its samples are coordinated

Phase 2 showed BPS + ROS fails. Sampling from BPS's target with Array-RQMC, scoring with per-decision importance sampling (PDIS), and merging prefix weights works.

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

- **Against the original BPS methods:** it is 1.5–70× better than BPS + OIS on every domain. It is 1.5–54× better than BPS + ROS at n = 8192; on πe-A, BPS + ROS is still better at small n. It stays unbiased (|bias|/SE ≤ 1.4).
- **Against ROS-MC:**
  - **Noisy domains:** it matches or beats ROS-MC per episode (significant on slip 0.2 and the noisy trap).
  - **Deterministic domains:** ROS-MC still wins, by 7–20× with Q-ordering. Its rate there is n⁻¹·⁸, against ROS-MC's n⁻².
- **Why it works:**
  - Coordination removes the variance of which action is drawn in each state.
  - PDIS stops later weights from multiplying earlier rewards.
  - Merging removes the spread of weights among episodes that reach the same state. With merging, the weighted estimate gets close to the model-based (MB) estimate on the same data: 0.335 vs 0.326 on slip 0.2.
- **A target built for coordination doesn't help.** Minimizing the variance left after coordination (J, `core/j_target.py`) gives a target within 3.5–6% of BPS's, and the measured difference is not significant (`results/j_target.md`). J itself predicts the measured variance within 0–10%.

### 2. Smooth continuous actions: QMC beats ROS and PROPS

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

### 3. Unbiased estimates with valid error bars

Data collected with ROS or PROPS give biased Monte Carlo estimates, because each action depends on all earlier data. When their variance is small, that bias makes confidence intervals undercover. The RQMC designs keep every trajectory exactly distributed as the sampling policy.

| | bias/SE | coverage of 95% intervals from 10 independent runs |
|---|---|---|
| ROS, deterministic GridWorld | −12 to −20 | 55–78% |
| ROS / PROPS, LQ (1-D, H = 10, σ_w ≤ 0.1) | −5 to −14 | 76–92% |
| Sobol RQMC and Array-RQMC (all domains) | within ±2 except one cell (+3.1) | 88–99% |

### Where each method wins
- **Discrete actions, on-policy:** ROS. Its adaptive per-state balancing reaches n⁻² in the deterministic GridWorld.
- **Smooth continuous actions:** QMC.
- **BPS:** combine it with coordinated sampling (Array-RQMC, PDIS, merged weights), not with ROS.

**Tried, neutral or negative:**
- Per-state QMC is a weaker, non-adaptive ROS: with ROS's α tuned at each n, it wins only on πe-A at n ≤ 256. With BPS's weights it blows up.
- Latin hypercube and antithetic sampling are weaker than Sobol. Antithetic is 2× worse than i.i.d. in LQ, whose returns are symmetric in the action noise.
- Adding RQMC on top of ROS gives nothing.
- On-policy in the discrete GridWorld, ROS beats every QMC sampler.
- Per environment step, coordinated BPS loses to ROS-MC, because BPS's target lengthens episodes (25–27 steps vs 19–21).
- Q-ordering used the exact Q (a best case); in practice Q would have to be estimated.

## Run

From the repository root:

```bash
python -m phase2_5_qmc.experiments.gridworld    # 6 min: QMC samplers vs ROS on the GridWorld (+ figure)
python -m phase2_5_qmc.experiments.array_rqmc   # 4 min: Array-RQMC, bias and interval coverage
python -m phase2_5_qmc.experiments.lq           # 1 hour: continuous actions, QMC vs ROS vs PROPS
python -m phase2_5_qmc.experiments.j_target     # 8 min: a target designed for coordination
python -m phase2_5_qmc.experiments.merging      # 8 min: prefix-weight merging and Q-ordered actions
```

## Code

```
core/qmc.py            per-state QMC and trajectory-level designs (Sobol, Latin hypercube, antithetic)
core/array_rqmc.py     Array-RQMC in lockstep, with weight-aware sorting, prefix-weight merging, Q-ordering
core/lq.py             linear-quadratic testbed (exact value) and its samplers, including ROS and PROPS
core/j_target.py       the variance left after coordination (J), its exact gradient and its minimizer
```

Environments, exact DP, ROS toward πe and BPS's target come from Phases 1 and 2.

**References.**
- S. M. R. Arnold, P. L'Ecuyer, L. Chen, Y. Chen, F. Sha. *Policy Learning and Evaluation with Randomized Quasi-Monte Carlo.* AISTATS 2022.
- P. L'Ecuyer, C. Lécot, B. Tuffin. *A Randomized Quasi-Monte Carlo Simulation Method for Markov Chains.* Operations Research 2008.
- D. Precup, R. S. Sutton, S. Singh. *Eligibility Traces for Off-Policy Policy Evaluation.* ICML 2000 (per-decision IS).
- N. E. Corrado, J. P. Hanna. *On-Policy Policy Gradient Reinforcement Learning Without On-Policy Sampling* (PROPS).
