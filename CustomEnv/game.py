import numpy as np
import cv2
import random
from gym import Env, spaces

from CustomEnv.objects import FireBall, Carrot, Pegasus

font = cv2.FONT_HERSHEY_SIMPLEX


class PegasusEnv(Env):
    def __init__(self):
        super(PegasusEnv, self).__init__()

        self.energy_count = None
        self.pegasus = None
        self.carrot_count_spawned = 0
        self.fire_count = None
        self.ep_return = None
        self.energy_left = None

        # Define a 2-D observation space

        self.observation_shape = (600, 800, 3)
        self.observation_space = spaces.Box(low=np.zeros(self.observation_shape),
                                            high=np.ones(self.observation_shape),
                                            dtype=np.float16)

        # Define an action space ranging from 0 to 4
        self.action_space = spaces.Discrete(5, )

        # Create a canvas to render the environment images upon
        self.canvas = np.ones(self.observation_shape) * 1

        # Define elements present inside the environment
        self.elements = []

        self.max_energy = 1000
        self.carrots_count = 0

        # Permissible area of pegasus to be
        self.y_min = int(self.observation_shape[0] * 0.1)
        self.x_min = 0
        self.y_max = int(self.observation_shape[0] * 0.9)
        self.x_max = self.observation_shape[1]

    def reset(self):
        self.energy_left = self.max_energy
        self.ep_return = 0
        self.fire_count = 0
        self.energy_count = 0

        # Determine a place to initialise the pegasus in
        x = random.randrange(int(self.observation_shape[0] * 0.05), int(self.observation_shape[0] * 0.10))
        y = random.randrange(int(self.observation_shape[1] * 0.15), int(self.observation_shape[1] * 0.20))

        # Initialise the pegasus
        self.pegasus = Pegasus("pegasus", self.x_max, self.x_min, self.y_max, self.y_min)
        self.pegasus.set_position(x, y)

        # Initialise the elements
        self.elements = [self.pegasus]

        # Reset the Canvas
        self.canvas = np.ones(self.observation_shape) * 1

        self.draw_elements_on_canvas()

        return self.canvas

    def draw_elements_on_canvas(self):
        # Init the canvas
        self.canvas = np.ones(self.observation_shape) * 1

        for elem in self.elements:
            elem_shape = elem.icon.shape
            x, y = elem.x, elem.y
            self.canvas[y: y + elem_shape[1], x:x + elem_shape[0]] = elem.icon

        text = 'Energy Left: {} | Rewards: {}'.format(self.energy_left, self.ep_return)
        self.canvas = cv2.putText(self.canvas, text, (10, 20), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

        # Check if the pegasus has taken a carrot
        carrot_text = 'Carrots Eaten: {}'.format(self.carrots_count)
        self.canvas = cv2.putText(self.canvas, carrot_text, (10, 50), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

        # Check if the game has ended
        if self.energy_left == 0 or self.pegasus not in self.elements:
            end_text = 'Game Over'
            self.canvas = cv2.putText(self.canvas, end_text,
                                      (self.observation_shape[1] // 2 - 50, self.observation_shape[0] // 2), font, 1,
                                      (0, 0, 255), 2, cv2.LINE_AA)

        # Check if the Pegasus has eaten three carrots and won the game
        if self.carrots_count == 3:
            self.canvas = cv2.putText(self.canvas, "Game Won!",
                                      (self.observation_shape[1] // 2 - 50, self.observation_shape[0] // 2),
                                      font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    def render(self, mode="human"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""

        if mode == "human":
            cv2.imshow("Game", self.canvas)
            cv2.waitKey(10)

            if self.energy_left == 0 or self.pegasus not in self.elements or self.carrots_count == 3:
                cv2.waitKey(0)  # Wait for a key press to close the window

        elif mode == "rgb_array":
            return self.canvas

    def close(self):
        cv2.destroyAllWindows()

    @staticmethod
    def get_action_meanings():
        return {0: "Right", 1: "Left", 2: "Down", 3: "Up", 4: "Do Nothing"}

    @staticmethod
    def has_collided(elem1, elem2):
        x_col = False
        y_col = False

        elem1_x, elem1_y = elem1.get_position()
        elem2_x, elem2_y = elem2.get_position()

        if 2 * abs(elem1_x - elem2_x) <= (elem1.icon_w + elem2.icon_w):
            x_col = True

        if 2 * abs(elem1_y - elem2_y) <= (elem1.icon_h + elem2.icon_h):
            y_col = True

        if x_col and y_col:
            return True

        return False

    def step(self, action):
        done = False

        # Assert that it is a valid action
        assert self.action_space.contains(action), "Invalid Action"

        self.energy_left -= 1
        reward = 0

        # apply the action to the pegasus
        if action == 0:
            self.pegasus.move(0, 5)
        elif action == 1:
            self.pegasus.move(0, -5)
        elif action == 2:
            self.pegasus.move(5, 0)
        elif action == 3:
            self.pegasus.move(-5, 0)
        elif action == 4:
            self.pegasus.move(0, 0)

        # Spawn a fire at the right edge with prob 0.009
        if random.random() < 0.009:
            spawned_fire = FireBall("fire_{}".format(self.fire_count), self.x_max, self.x_min, self.y_max, self.y_min)
            self.fire_count += 1

            # Compute the x,y co-ordinates of the position from where the fire has to be spawned
            # Horizontally, the position is on the right edge and vertically, the height is randomly
            # sampled from the set of permissible values
            fire_x = self.x_max
            fire_y = random.randrange(self.y_min, self.y_max)
            spawned_fire.set_position(fire_x, fire_y)
            self.elements.append(spawned_fire)

        # Spawn a carrot at the bottom edge with probability
        if random.random() < 0.03:
            spawned_carrot = Carrot("carrot_{}".format(self.carrot_count_spawned),
                                    self.x_max, self.x_min, self.y_max, self.y_min)
            self.carrot_count_spawned += 1

            # Compute the x,y co-ordinates of the position from where the energy tank has to be spawned
            # Horizontally, the position is randomly chosen from the list of permissible values and
            # vertically, the position is on the bottom edge
            energy_x = random.randrange(self.x_min, self.x_max)
            energy_y = self.y_max
            spawned_carrot.set_position(energy_x, energy_y)
            self.elements.append(spawned_carrot)

        for elem in self.elements:
            if isinstance(elem, FireBall):
                # If the fire has reached the left edge, remove it from the Env
                if elem.get_position()[0] <= self.x_min:
                    self.elements.remove(elem)
                else:
                    # Move the fire left by 5 pts.
                    elem.move(-5, 0)

                # If the fire has collided.
                if self.has_collided(self.pegasus, elem):
                    done = True
                    reward = -10
                    self.elements.remove(self.pegasus)

            if isinstance(elem, Carrot):
                # If the carrot has reached the top, remove it from the Env
                if elem.get_position()[1] <= self.y_min:
                    self.elements.remove(elem)
                else:
                    # Move the carrot up by 5 pts.
                    elem.move(0, -5)

                # If the carrot has collided with the pegasus.
                if self.has_collided(self.pegasus, elem):
                    # Remove the carrot from the env.
                    self.elements.remove(elem)
                    self.carrots_count += 1
                    reward += 10

                    if self.carrots_count == 3:
                        done = True

                    self.energy_left += 100

        self.ep_return += 1
        self.draw_elements_on_canvas()

        if self.energy_left == 0:
            done = True

        return self.canvas, reward, done, []


if __name__ == "__main__":
    env = PegasusEnv()
    obs = env.reset()

    while True:
        # Take a random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        env.render()
        if done:
            break

    env.close()
