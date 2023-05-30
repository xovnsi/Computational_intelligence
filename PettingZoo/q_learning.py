from matplotlib import pyplot as plt
from pathlib import Path
from typing import Union, Any

import numpy as np
import pettingzoo
from numpy import ndarray


def plot_learning_curve(learning_curve, filename):
    plt.plot(learning_curve)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'{filename} Learning Curve')
    plt.savefig(f'{filename}.png')
    plt.show()

class QLearner:
    def __init__(
            self,
            action_space: pettingzoo.utils.wrappers.base,
            agent: str,
            gamma: float = 0.99,
            lr: float = 0.01,
            eps_init: float = 0.5,
            eps_min: float = 1e-5,
            eps_step: float = 1e-3,
            name: str = "Q-learning",
    ):

        self.q_values = {}
        self.action_space: pettingzoo.utils.wrappers.base = action_space
        self.agent: str = agent
        self.gamma: float = gamma
        self.eps: float = eps_init
        self.eps_min: float = eps_min
        self.eps_step: float = eps_step
        self.lr: float = lr

        self.models_dir: Path = Path("players")
        self.models_dir.mkdir(exist_ok=True, parents=True)
        self.player_path: str = f"{self.models_dir}/{name}"
        self.reset()

    def get_best_action(self, obs):
        obs = str(obs["observation"])
        if obs not in self.q_values:
            self.q_values[obs] = np.zeros(9)
        return np.argmax(self.q_values[obs])

    def eps_greedy(self, obs: dict, eps: Union[float, None] = None) -> ndarray[int] | Any:
        eps = eps or self.eps
        if np.random.random() < eps:
            return self.action_space(self.agent).sample(mask=obs["action_mask"])
        else:
            return self.get_best_action(obs)

    def get_action(self, obs: dict, eps: Union[float, None] = None) -> int:
        return self.eps_greedy(obs, eps)

    def get_random(self, obs: dict) -> int:
        return self.action_space(self.agent).sample(mask=obs["action_mask"])

    def epsilon_decay(self) -> None:
        self.eps = max(self.eps - self.eps_step, self.eps_min)

    def update(self, obs, action, reward, terminated, next_obs):
        next_obs = str(next_obs["observation"])
        obs = str(obs["observation"])
        estimate_value_at_next_state = (not terminated) * np.max(
            self.q_values.get(next_obs, np.zeros(9))
        )
        new_estimate = reward + self.gamma * estimate_value_at_next_state

        if obs not in self.q_values:
            self.q_values[obs] = np.zeros(9)

        self.q_values[obs][action] = (1 - self.lr) * self.q_values[obs][action] + self.lr * new_estimate
        self.epsilon_decay()

    def reset(self):
        self.q_values = {}
