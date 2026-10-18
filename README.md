# Behavior policy search + robust on-policy sampling for policy evaluation

Rohan Gupta · CS 799 with Prof. Josiah Hanna · UW–Madison, Fall 2026

**Question.** Estimating a policy's value from its own trajectories is noisy for two reasons:
1. The return varies a lot from one trajectory to the next.
2. A finite sample of actions deviates from the policy by chance.

Behavior policy search (BPS, [1]) attacks the first: it collects data with a lower-variance behavior policy and reweights it. Robust on-policy sampling (ROS, [2]) attacks the second: it steers each action so that the empirical action frequencies track the policy. Can the two be combined to beat both?

Each directory is one phase. Every number in a phase's README comes from the scripts in that directory, and all scripts run from the repository root with `python -m`.

| Phase | Result |
|---|---|
| [1. Reproduction](phase1_reproduction/) | Both papers' results reproduce on the ROS repository's GridWorld and MultiBandit. We also identify why BPS looked no better than on-policy sampling in [2]. |
| [2. BPS + ROS](phase2_bps_plus_ros/) | **Negative.** Tracking the BPS target with ROS does not beat ROS: ROS-MC wins by 2× to 2500× in GridWorld, and in the bandit the combination never beats both parents. Two reasons: BPS optimizes the wrong objective once ROS removes action-sampling variance, and per-state tracking cannot control trajectory-level importance weights. |

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
