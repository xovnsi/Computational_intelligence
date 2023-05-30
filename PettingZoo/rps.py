import random
import numpy as np
from pettingzoo.classic import rps_v2
import matplotlib.pyplot as plt

epsilon = 0.1
alpha = 0.1
gamma = 0.8
q_table = np.zeros((4, 3))

# Create environment and show the game later
env = rps_v2.env(render_mode="")


# Start game and get initial observation
env.reset()
state, reward, terminated, truncated, info = env.last()


obs = env.step(0)
episode_list = []
reward_list = []

# Set up loop for running Q-learning
for i_episode in range(10000):
    env.reset()
    obs = env.step(0)

    total_reward = 0
    agent_1_reward = 0
    agent_0_reward = 0

    for agent in env.agent_iter():
        # env.render()

        if state not in q_table:
            q_table[state] = [0, 0, 0]

        if agent == 'player_0':
            if random.uniform(0, 1) < epsilon:
                action = random.choice([0, 1, 2])
            else:
                action = np.argmax(q_table[state])
        else:
            action = random.choice([0, 1, 2])

        next_state, reward, terminated, truncated, info = env.last()

        if terminated or truncated:
            break

        if agent == 'player_1':
            agent_1_reward += reward

        if agent == 'player_0':
            agent_0_reward += reward

        env.step(action)

        # next_state = obs
        if next_state not in q_table:
            q_table[next_state] = [0, 0, 0]

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

        q_table[state, action] = new_value

        state = next_state

    episode_list.append(i_episode)
    reward_list.append(agent_0_reward)

    print(f'agent_0_reward: {agent_0_reward}, agent_1_reward: {agent_1_reward}')
    print("Episode: {}, Total Reward: {}".format(i_episode, total_reward))

env.close()


plt.plot(episode_list, reward_list)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Learning Curve")
plt.show()


# test the trained agent
# env.reset()

test_episode_num = 1

for i_episode in range(test_episode_num):
    env.reset()
    obs = env.step(0)

    test_reward = 0
    test_player0 = 0
    test_player1 = 0

    for agent in env.agent_iter():
        # env.render()

        state = obs
        action = np.argmax(q_table[state])

        obs, reward, terminated, truncated, info = env.last()

        if terminated or truncated:
            break

        print(f'agent: {agent}, action: {action}, reward: {reward}')

        if action not in env.action_spaces[agent]:
            continue

        test_reward += reward

        if agent == 'player_0':
            test_player0 += reward

        if agent == 'player_1':
            test_player1 += reward

        env.step(action)

    print("Test Reward: {}".format(test_reward))
    print("Test Player 0 Reward: {}".format(test_player0))
    print("Test Player 1 Reward: {}".format(test_player1))

env.close()



