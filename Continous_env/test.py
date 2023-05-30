import gymnasium as gym
from stable_baselines3 import A2C

env = gym.make('LunarLander-v2')  # continuous: LunarLanderContinuous-v2
env.reset()

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

episodes = 5

for ep in range(episodes):
    obs = env.reset()
    terminated = False
    truncated = False
    while not terminated and not truncated:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        print(rewards)
