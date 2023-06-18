import gym
import pandas as pd
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

env = gym.make('Pendulum-v1')

hyperparams = [
    {'learning_rate': 0.01, 'batch_size': 128, 'gamma': 0.8},
    {'learning_rate': 0.001, 'batch_size': 128, 'gamma': 0.9},
    {'learning_rate': 0.1, 'batch_size': 128, 'gamma': 0.99},
]

results = []

for params in hyperparams:
    rewards = []
    stds = []

    # Powtórz eksperyment 10 razy
    for _ in range(10):
        vec_env = make_vec_env('Pendulum-v1', n_envs=1)
        model = PPO('MlpPolicy', vec_env, **params, verbose=1)
        logger = configure("results/", ["csv"])
        model.set_logger(logger)
        model.learn(total_timesteps=50000)

        output = pd.read_csv("results/progress.csv", sep=',')
        ep_rew = output['rollout/ep_rew_mean'].to_numpy()
        rewards.append(np.mean(ep_rew))
        stds.append(np.std(ep_rew))

    results.append((rewards, stds))

# krzywe uczenia
fig, axs = plt.subplots(len(hyperparams), figsize=(8, 6))
for i, (params, (rewards, stds)) in enumerate(zip(hyperparams, results)):
    axs[i].plot(rewards, label='Reward')
    axs[i].fill_between(range(len(rewards)), np.array(rewards) - np.array(stds), np.array(rewards) + np.array(stds),
                        alpha=0.3)
    axs[i].set_title(f'Hyperparams: {params}')
    axs[i].set_xlabel('Step/Episode')
    axs[i].set_ylabel('Reward')
    axs[i].legend()
plt.tight_layout()
plt.show()
