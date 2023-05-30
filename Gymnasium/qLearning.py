import random
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CliffWalking-v0')

num_episodes = 100
learning_rate = 0.2
discount_factor = 0.90
epsilon = 0.2

# Initialize the Q-table
num_states = env.observation_space.n
num_actions = env.action_space.n
q_table = np.zeros((num_states, num_actions))

# Initialize the lists for the learning curve
reward_list = []
episode_list = []


# Run the Q-learning algorithm
def q_learning(num_episodes, learning_rate, discount_factor):
    for episode in range(num_episodes):
        state = env.reset()[0]
        total_reward = 0
        terminated = truncated = False

        while not terminated and not truncated:
            # Choose an action
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(q_table[state])  # Exploit learned values

            # Take the chosen action and observe the outcome
            next_state, reward, terminated, truncated, info = env.step(action)
            # print(f'Step {step}: observation={observation}, action={action}, state={state}, reward={reward};')

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)

            q_table[state, action] = new_value

            # Update the current state and reward
            state = next_state
            total_reward += reward

        # Append the episode number and total reward to the lists for the learning curve
        episode_list.append(episode)
        reward_list.append(total_reward)

    return q_table, episode_list, reward_list


q_table, episode_list, reward_list = q_learning(num_episodes, learning_rate, discount_factor)

# Plot the learning curve
plt.plot(episode_list, reward_list)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('CliffWalking Q-learning')
plt.show()


num_test_episodes = 10
# total_reward = 0

for episode in range(num_test_episodes):
    state = env.reset()[0]
    terminated = False
    truncated = False
    total_rewards = 0

    while not terminated and not truncated:
        action = np.argmax(q_table[state])
        state, reward, terminated, truncated, info = env.step(action)
        total_rewards += reward

    print(f'Episode {episode + 1}: total_rewards={total_rewards}')

# print('Average total reward over %d test episodes: %f' % (num_test_episodes, total_reward / num_test_episodes))
