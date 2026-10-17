"""The ROS repository's tabular environments, rebuilt exactly (checked in tests.py).

GridWorld (repo experiments/gridworld.py)
  * 4x4 grid, start (0, 0), state index 4x + y, actions Left, Right, Down, Up.
  * Rewards are paid on entering a cell: goal (3, 3) +10 and the episode ends, trap (1, 1) -10,
    bonus (1, 3) +1, any other move -1. Bumping into a wall also costs -1.
  * Horizon 100, gamma = 1.

MultiBandit (repo experiments/multibandit.py)
  * 30 arms, one fixed instance drawn with np.random.seed(0) (means, then scales, both U[0, 1)).
  * The reward of arm a is Normal(mean_a, scale_a^2).
"""
from dataclasses import dataclass

import numpy as np

from . import repo_io

LEFT, RIGHT, DOWN, UP = 0, 1, 2, 3
_MOVES = {LEFT: (-1, 0), RIGHT: (1, 0), DOWN: (0, -1), UP: (0, 1)}
_PERPENDICULAR = {LEFT: (DOWN, UP), RIGHT: (DOWN, UP), DOWN: (LEFT, RIGHT), UP: (LEFT, RIGHT)}
GOAL, TRAP, BONUS = 4 * 3 + 3, 4 * 1 + 1, 4 * 1 + 3


@dataclass
class TabularMDP:
    """Finite-horizon tabular MDP. Terminal states are absorbing and take no decisions."""
    name: str
    P: np.ndarray            # (S, A, S) transition probabilities
    R: np.ndarray            # (S, A, S) mean reward of each transition
    reward_std: np.ndarray   # (S, A, S) std of the Gaussian reward noise
    terminal: np.ndarray     # (S,) bool
    d0: np.ndarray           # (S,) initial-state distribution
    horizon: int             # maximum number of decisions per episode
    gamma: float = 1.0

    @property
    def S(self):
        return self.P.shape[0]

    @property
    def A(self):
        return self.P.shape[1]


def gridworld(slip=0.0, horizon=100):
    """The repo's GridWorld. With slip > 0, a move instead goes to each perpendicular
    direction with probability slip / 2 (the stochastic GridWorld of [1]; used only in tests)."""
    S, A = 16, 4

    def move(s, a):
        x, y = divmod(s, 4)
        dx, dy = _MOVES[a]
        return 4 * min(max(x + dx, 0), 3) + min(max(y + dy, 0), 3)

    def reward(s, s2):
        if s2 == s:                                    # bumped into a wall
            return -1.0
        return {GOAL: 10.0, TRAP: -10.0, BONUS: 1.0}.get(s2, -1.0)

    P = np.zeros((S, A, S))
    R = np.zeros((S, A, S))
    for s in range(S):
        if s == GOAL:
            P[s, :, s] = 1.0
            continue
        for a in range(A):
            outcomes = [(a, 1.0 - slip)] + ([(b, slip / 2) for b in _PERPENDICULAR[a]] if slip > 0 else [])
            for b, p in outcomes:
                P[s, a, move(s, b)] += p
            for s2 in np.nonzero(P[s, a])[0]:
                R[s, a, s2] = reward(s, s2)
    terminal = np.zeros(S, bool)
    terminal[GOAL] = True
    d0 = np.zeros(S)
    d0[0] = 1.0
    return TabularMDP("GridWorld" + (f"-slip{slip:g}" if slip else ""), P, R, np.zeros_like(R),
                      terminal, d0, horizon)


def bandit(means, stds, name="Bandit"):
    """One-step bandit: state 0 chooses an arm, state 1 is terminal."""
    means, stds = np.asarray(means, float), np.asarray(stds, float)
    A = means.shape[0]
    P = np.zeros((2, A, 2))
    P[:, :, 1] = 1.0
    R = np.zeros((2, A, 2))
    R[0, :, 1] = means
    reward_std = np.zeros((2, A, 2))
    reward_std[0, :, 1] = stds
    return TabularMDP(name, P, R, reward_std, np.array([False, True]), np.array([1.0, 0.0]), 1)


def multibandit():
    """The repo's fixed 30-arm bandit."""
    rs = np.random.RandomState(0)
    means = rs.rand(30)
    stds = rs.rand(30)
    return bandit(means, stds, "MultiBandit")


def repo_policy_logits(name, mdp):
    """The repo's pretrained policy model_<name>.pt as an (S, A) logit table for mdp."""
    table = repo_io.policy_logits(name)
    if table.shape == (mdp.S, mdp.A):
        return table
    logits = np.zeros((mdp.S, mdp.A))                  # terminal states take no decisions
    logits[~mdp.terminal] = table[: int((~mdp.terminal).sum())]
    return logits


def eps_greedy_logits(base_logits, eps, terminal):
    """The repo's epsilon-greedy policies (its GridWorld_k files have eps = k/10): the greedy
    action of base_logits gets probability 1 - eps + eps/A and every other action eps/A."""
    A = base_logits.shape[1]
    logits = np.zeros_like(base_logits)
    logits[np.arange(base_logits.shape[0]), base_logits.argmax(1)] = np.log((A - (A - 1) * eps) / eps)
    logits[terminal] = 0.0
    return logits
