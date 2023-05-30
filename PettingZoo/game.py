import numpy as np
from pettingzoo.classic import tictactoe_v3
from pettingzoo.utils import OrderEnforcingWrapper
from tqdm import tqdm
from q_learning import QLearner, plot_learning_curve


class Game:
    def __init__(
            self,
            env: OrderEnforcingWrapper,
            player_1: QLearner,
            player_2: QLearner,
    ) -> None:
        self.env = env
        self.player_1: QLearner = player_1
        self.player_2: QLearner = player_2

    def train(self, epoch: int, verbose: bool = True) -> None:

        learning_curve_0 = []
        learning_curve_1 = []

        nb_wins_agent_1: int = 0
        nb_wins_agent_2: int = 0
        nb_draws: int = 0

        agent_1_reward = 0
        agent_2_reward = 0

        for i in tqdm(range(epoch)):
            was_draw: bool = False
            self.env.reset()
            for agent in self.env.agent_iter():
                last_observation, reward, termination, truncation, info = self.env.last()

                if agent == "player_1":
                    agent_1_reward += reward

                if agent == "player_2":
                    agent_2_reward += reward

                if termination:
                    if reward == 1 and agent == "player_1":
                        nb_wins_agent_1 += 1
                    elif reward == 1 and agent == "player_2":
                        nb_wins_agent_2 += 1
                    elif reward == 0 and agent == "player_1":
                        was_draw = True
                    elif reward == 0 and agent == "player_2" and was_draw:
                        nb_draws += 1

                    if i % 500 == 0 and verbose and i != 0 and agent == "player_1":
                        print(
                            f"\nAgent 1 wins: {nb_wins_agent_1}, Agent 2 wins: {nb_wins_agent_2}, Draws: {nb_draws}, "
                            f"Ratio: {100 * (nb_wins_agent_1 / (nb_wins_agent_2 + nb_wins_agent_1)) : .2f}"
                        )

                    self.env.step(None)
                elif truncation:
                    if verbose:
                        print("Truncated")
                else:  # we update the actor and critic networks weights every steps
                    if agent == "player_1":
                        action = self.player_1.get_action(last_observation)
                    else:
                        # action = self.player_2.get_action(last_observation)
                        action = self.player_2.get_random(last_observation)

                    while last_observation['action_mask'][action] == 0:
                        if agent == "player_1":
                            action = self.player_1.get_action(last_observation)
                        else:
                            # action = self.player_2.get_action(last_observation)
                            action = self.player_2.get_random(last_observation)

                    self.env.step(action)
                    observation, reward, termination, truncation, info = self.env.last()

                    if agent == "player_1":
                        self.player_1.update(
                            last_observation, action, reward, termination, observation
                        )
                    else:
                        self.player_2.update(
                            last_observation, action, reward, termination, observation
                        )

            learning_curve_0.append(agent_1_reward)
            learning_curve_1.append(agent_2_reward)

        plot_learning_curve(learning_curve_0, "q table")

    def eval(self, nb_eval: int, verbose: int = 0):
        termination: bool = False
        truncation: bool = False

        nb_wins_agent_1: int = 0
        nb_wins_agent_2: int = 0
        nb_draws: int = 0
        i: int = 0

        while (nb_wins_agent_1 + nb_wins_agent_2 + nb_draws) < nb_eval:
        # while not termination or not truncation:
            self.env.reset()
            was_draw: bool = False

            for agent in self.env.agent_iter():
                last_observation, reward, termination, truncation, info = self.env.last()

                if termination:

                    if reward == 1 and agent == "player_1":
                        nb_wins_agent_1 += 1
                        i += 1
                    elif reward == 1 and agent == "player_2":
                        nb_wins_agent_2 += 1
                        i += 1
                    elif reward == 0 and agent == "player_1":
                        was_draw = True
                    elif reward == 0 and agent == "player_2" and was_draw:
                        nb_draws += 1
                        i += 1

                    self.env.step(None)

                elif truncation:
                    if verbose:
                        print("Truncated")
                else:
                    if agent == "player_1":
                        action = self.player_1.get_action(last_observation)
                    else:
                        action = self.player_2.get_action(last_observation)

                    while last_observation['action_mask'][action] == 0:
                        if agent == "player_1":
                            action = self.player_1.get_action(last_observation)
                        else:
                            action = self.player_2.get_action(last_observation)

                    self.env.step(action)
        print(f"Agent1 wins: {nb_wins_agent_1}\nAgent2 wins: {nb_wins_agent_2}\nDraws: {nb_draws}")


if __name__ == "__main__":
    N_EPOCH = 100
    N_EVAL = 10

    env: OrderEnforcingWrapper = tictactoe_v3.env()
    space = env.action_space
    player_1 = QLearner(space, "player_1", name="player_1")
    player_2 = QLearner(space, "player_2", name="player_2")

    game = Game(env, player_1, player_2)
    game.train(epoch=N_EPOCH, verbose=True)
    game.eval(nb_eval=N_EVAL, verbose=True)

