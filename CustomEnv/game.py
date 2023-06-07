import numpy as np
import cv2
import random
from gym import Env, spaces

font = cv2.FONT_HERSHEY_SIMPLEX


class PegasusScape(Env):
    def __init__(self):
        super(PegasusScape, self).__init__()

        # Define a 2-D observation space
        self.observation_shape = (600, 800, 3)
        self.observation_space = spaces.Box(low=np.zeros(self.observation_shape),
                                            high=np.ones(self.observation_shape),
                                            dtype=np.float16)

        # Define an action space ranging from 0 to 4
        self.action_space = spaces.Discrete(6, )

        # Create a canvas to render the environment images upon
        self.canvas = np.ones(self.observation_shape) * 1

        # Define elements present inside the environment
        self.elements = []

        # Maximum energy pegasus can take at once
        self.max_energy = 1000
        self.carrots_eaten = 0

        # Permissible area of pegasus to be
        self.y_min = int(self.observation_shape[0] * 0.1)
        self.x_min = 0
        self.y_max = int(self.observation_shape[0] * 0.9)
        self.x_max = self.observation_shape[1]

    def reset(self):
        # Reset the energy consumed
        self.energy_left = self.max_energy

        # Reset the reward
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

        # Draw elements on the canvas
        self.draw_elements_on_canvas()

        # return the observation
        return self.canvas

    def draw_elements_on_canvas(self):
        # Init the canvas
        self.canvas = np.ones(self.observation_shape) * 1

        # Draw the pegasus on canvas
        for elem in self.elements:
            elem_shape = elem.icon.shape
            x, y = elem.x, elem.y
            self.canvas[y: y + elem_shape[1], x:x + elem_shape[0]] = elem.icon

        text = 'Energy Left: {} | Rewards: {}'.format(self.energy_left, self.ep_return)

        # Put the info on canvas
        self.canvas = cv2.putText(self.canvas, text, (10, 20), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

        # Check if the pegasus has taken a carrot
        if self.energy_count > 0:
            carrot_text = 'Carrots Eaten: {}'.format(self.carrots_eaten)
            self.canvas = cv2.putText(self.canvas, carrot_text, (10, 50), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

        # Check if the game has ended
        if self.energy_left == 0 or self.pegasus not in self.elements:
            end_text = 'Game Over'
            self.canvas = cv2.putText(self.canvas, end_text,
                                      (self.observation_shape[1] // 2 - 50, self.observation_shape[0] // 2), font, 1,
                                      (0, 0, 255), 2, cv2.LINE_AA)

    def render(self, mode="human"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            cv2.imshow("Game", self.canvas)
            cv2.waitKey(10)
            if self.energy_left == 0 or self.pegasus not in self.elements:
                cv2.waitKey(0)  # Wait for a key press to close the window

        elif mode == "rgb_array":
            return self.canvas

    def close(self):
        cv2.destroyAllWindows()

    def get_action_meanings(self):
        return {0: "Right", 1: "Left", 2: "Down", 3: "Up", 4: "Do Nothing"}

    def has_collided(self, elem1, elem2):
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
        # Flag that marks the termination of an episode
        done = False

        # Assert that it is a valid action
        assert self.action_space.contains(action), "Invalid Action"

        # Decrease the energy counter
        self.energy_left -= 1

        # Reward for executing a step.
        reward = 1

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

        # Spawn a fire at the right edge with prob 0.01
        if random.random() < 0.01:
            # Spawn a fire
            spawned_fire = FireBall("fire_{}".format(self.fire_count), self.x_max, self.x_min, self.y_max, self.y_min)
            self.fire_count += 1

            # Compute the x,y co-ordinates of the position from where the fire has to be spawned
            # Horizontally, the position is on the right edge and vertically, the height is randomly
            # sampled from the set of permissible values
            fire_x = self.x_max
            fire_y = random.randrange(self.y_min, self.y_max)
            spawned_fire.set_position(self.x_max, fire_y)

            # Append the spawned fire to the elements currently present in Env.
            self.elements.append(spawned_fire)

            # Spawn a energy at the bottom edge with prob 0.01
        if random.random() < 0.01:
            # Spawn a energy tank
            spawned_energy = Energy("energy_{}".format(self.fire_count), self.x_max, self.x_min, self.y_max, self.y_min)
            self.energy_count += 1

            # Compute the x,y co-ordinates of the position from where the energy tank has to be spawned
            # Horizontally, the position is randomly chosen from the list of permissible values and
            # vertically, the position is on the bottom edge
            energy_x = random.randrange(self.x_min, self.x_max)
            energy_y = self.y_max
            spawned_energy.set_position(energy_x, energy_y)

            # Append the spawned energy tank to the elemetns currently present in the Env.
            self.elements.append(spawned_energy)

            # For elements in the Ev
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
                    # Conclude the episode and remove the pegasus from the Env.
                    done = True
                    reward = -10
                    self.elements.remove(self.pegasus)

            if isinstance(elem, Energy):
                # If the energy tank has reached the top, remove it from the Env
                if elem.get_position()[1] <= self.y_min:
                    self.elements.remove(elem)
                else:
                    # Move the Tank up by 5 pts.
                    elem.move(0, -5)

                # If the carrot has collided with the pegasus.
                if self.has_collided(self.pegasus, elem):
                    # Remove the carrot from the env.
                    self.elements.remove(elem)
                    self.carrots_eaten += 1

                    # Fill the energy tank of the pegasus to full.
                    self.energy_left = self.max_energy

        # Increment the episodic return
        self.ep_return += 1

        # Draw elements on the canvas
        self.draw_elements_on_canvas()

        # If out of energy, end the episode.
        if self.energy_left == 0:
            done = True

        return self.canvas, reward, done, []


class Point(object):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        self.x = 0
        self.y = 0
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.name = name

    def set_position(self, x, y):
        self.x = self.clamp(x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(y, self.y_min, self.y_max - self.icon_h)

    def get_position(self):
        return (self.x, self.y)

    def move(self, del_x, del_y):
        self.x += del_x
        self.y += del_y

        self.x = self.clamp(self.x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(self.y, self.y_min, self.y_max - self.icon_h)

    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)


class Pegasus(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Pegasus, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("pegasus1.png") / 255.0
        self.icon_w = 128
        self.icon_h = 128
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))


class FireBall(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(FireBall, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("fireball.png") / 255.0
        self.icon_w = 64
        self.icon_h = 64
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))


class Energy(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Energy, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("carrot.png") / 255.0
        self.icon_w = 64
        self.icon_h = 64
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))


if __name__ == "__main__":
    env = PegasusScape()
    obs = env.reset()

    while True:
        # Take a random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        # Render the game
        env.render()
        if done:
            break

    env.close()
