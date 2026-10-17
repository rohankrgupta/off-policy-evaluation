# BPS reproduction: BPG-V on the GridWorld

## 1. Variance reduction of the learned behavior policy ([1]'s protocol)

BPG-V: batch 100, unit-norm step 0.1, baseline, 1000 iterations (10⁵ trajectories), 200 seeds. Reduction = 1 − Var_IS(θ)/Var(g), the MSE drop of fresh trajectories from behavior policy θ versus on-policy ones (exact). [1] reports 73.5% and 64.6% for its two GridWorld policies. Best possible: the optimal tabular behavior policy. BPG-V: the policy after 1000 iterations, mean over seeds [95% CI].

| πe | Var(g) | best possible | BPG-V | share of best |
|---|---|---|---|---|
| πe-A (GridWorld_5000) | 21.4 | 12.6% | 11.3% [11.2%, 11.5%] | 90% |
| ε-greedy 0.5 | 87.8 | 42.9% | 42.5% [42.4%, 42.5%] | 99% |
| ε-greedy 0.6 | 159.0 | 59.0% | 58.5% [58.5%, 58.5%] | 99% |
| πe-B (ε-greedy 0.7) | 315.8 | 70.5% | 69.8% [69.7%, 69.8%] | 99% |

## 2. MSE of BPS's own estimate relative to on-policy Monte Carlo

Adaptive OIS over all n trajectories collected while BPG-V learns: MSE / (Var(g)/n), mean ± SE over 200 seeds. Below 1 means BPS beats on-policy Monte Carlo with the same number of trajectories.

| πe | n = 100 | n = 1,000 | n = 10,000 | n = 100,000 |
|---|---|---|---|---|
| πe-A (GridWorld_5000) | 0.98 ± 0.09 | 0.91 ± 0.08 | 0.93 ± 0.10 | 0.86 ± 0.08 |
| ε-greedy 0.5 | 1.15 ± 0.11 | 0.80 ± 0.08 | 0.64 ± 0.07 | 0.54 ± 0.05 |
| ε-greedy 0.6 | 0.99 ± 0.09 | 0.73 ± 0.08 | 0.44 ± 0.04 | 0.39 ± 0.04 |
| πe-B (ε-greedy 0.7) | 0.95 ± 0.10 | 0.60 ± 0.05 | 0.34 ± 0.04 | 0.30 ± 0.03 |

## 3. Why BPS looked no better than on-policy sampling in [2]

[2]'s budget (8192 trajectories, adaptive OIS over all of them), 1000 seeds. [2]'s BPG is the repo's BehaviorPolicyGradient: plain SGD on −IS²·Σ log π with no baseline, rewards divided by v and a first batch of 9. A seed diverges when its final behavior policy has Var_IS > 2·Var(g).

| πe | BPG settings | MSE / MSE(OS-MC) [95% CI] | median final Var_IS / Var(g) | seeds diverged |
|---|---|---|---|---|
| πe-A | [2]: batch 10, SGD lr 0.01, rewards / v | 2.60 [0.96, 5.63] | 0.894 | 3.2% |
| πe-A | [1]: batch 100, unit-norm step 0.1, baseline | 0.95 [0.87, 1.03] | 0.894 | 0.0% |
| πe-B | [2]: batch 10, SGD lr 0.01, rewards / v | 7.23 [0.30, 17.79] | 0.305 | 0.2% |
| πe-B | [1]: batch 100, unit-norm step 0.1, baseline | 0.35 [0.32, 0.38] | 0.302 | 0.0% |
