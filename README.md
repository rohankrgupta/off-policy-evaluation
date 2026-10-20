# Behavior policy search + robust on-policy sampling for policy evaluation

Rohan Gupta · CS 799 with Prof. Josiah Hanna · UW–Madison, Fall 2026

**Question.** Estimating a policy's value from its own trajectories is noisy for two reasons:
1. The return varies a lot from one trajectory to the next.
2. A finite sample of actions deviates from the policy by chance.

Behavior policy search (BPS, [1]) attacks the first: it collects data with a lower-variance behavior policy and reweights it. Robust on-policy sampling (ROS, [2]) attacks the second: it steers each action so that the empirical action frequencies track the policy. Can the two be combined to beat both?

Each directory is one phase. Every result in a phase's README comes from the scripts in that directory, except lines explicitly marked as earlier experiments. All scripts run from the repository root with `python -m`.

| Phase | Result |
|---|---|
| [1. Reproduction](phase1_reproduction/) | Both papers' results reproduce on the ROS repository's GridWorld and MultiBandit. We also identify why BPS looked no better than on-policy sampling in [2]. |
| [2. BPS + ROS](phase2_bps_plus_ros/) | **Negative.** Tracking the BPS target with ROS does not beat ROS: ROS-MC wins by 2× to 2500× in GridWorld, and in the bandit the combination never beats both parents. Two reasons: BPS optimizes the wrong objective once ROS removes action-sampling variance, and per-state tracking cannot control trajectory-level importance weights. |
| [2.5. QMC with ROS and BPS](phase2_5_qmc/) | Randomized QMC samplers of the agent's actions (after Arnold et al., 2022). BPS works once its samples are coordinated: Array-RQMC with per-decision IS and merged weights is 1.5–70× better than BPS + OIS, unbiased, and matches or beats ROS-MC per episode with noise (ROS still wins in deterministic GridWorld). With smooth continuous actions, QMC beats ROS and PROPS by up to 185× and 670×. |
| [3. Model-based allocation](phase3_mb_allocation/) | A model-based estimator plus a behavior chosen to minimize its variance per environment step: a ReVar-style allocation with unknown transitions, general MDPs and step budgets. At equal steps it beats ROS-MC by 1.15× and 1.6× where noise is large or uneven, and ties otherwise. ROS adds nothing on top. |

## Setup

```bash
pip install -r requirements.txt
git clone https://github.com/uoe-agents/robust_onpolicy_data_collection external/robust_onpolicy_data_collection
git -C external/robust_onpolicy_data_collection checkout 056cebc
```

The ROS repository supplies the pretrained evaluation policies and its stored per-seed results. We read both without PyTorch and never modify them. Tested with Python 3.9, NumPy 2.0, SciPy 1.13, numba 0.60 and matplotlib 3.9.

## References
[1] J. P. Hanna, Y. Chandak, P. S. Thomas, M. White, P. Stone, S. Niekum. *Data-Efficient Policy Evaluation Through Behavior Policy Search.* JMLR 25, 2024.
[2] R. Zhong, D. Zhang, L. Schäfer, S. V. Albrecht, J. P. Hanna. *Robust On-Policy Sampling for Data-Efficient Policy Evaluation in Reinforcement Learning.* NeurIPS 2022.
