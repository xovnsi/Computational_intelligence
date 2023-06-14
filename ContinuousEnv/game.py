import gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np

# Utwórz środowisko
env = gym.make('Pendulum-v1')

# Definiuj zestawy hiperparametrów
hyperparams = [
    {'learning_rate': 0.01, 'batch_size': 32},
    {'learning_rate': 0.01, 'batch_size': 64},
    {'learning_rate': 0.1, 'batch_size': 128}
]

# Przechowuj wyniki
results = []

# Dla każdego zestawu hiperparametrów
for params in hyperparams:
    # Inicjalizuj puste listy na wyniki
    rewards = []
    stds = []

    # Powtórz eksperyment 10 razy
    for _ in range(10):
        # Utwórz model PPO z danymi hiperparametrami
        model = PPO('MlpPolicy', env, **params)

        # Trenuj model przez 50 000 kroków czasowych
        model.learn(total_timesteps=50000)

        # Testowanie nauczonego modelu przez 10 epizodów
        all_rewards = []
        for _ in range(10):
            obs = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
            all_rewards.append(episode_reward)

        # Dodaj wyniki do listy
        rewards.append(np.mean(all_rewards))
        stds.append(np.std(all_rewards))

    # Dodaj wyniki do głównej listy
    results.append((rewards, stds))

# Narysuj krzywe uczenia
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
