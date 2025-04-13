import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self, size=5, goal=(4,4), num_obstacles=3):
        self.size = size
        self.goal = goal
        self.num_obstacles = num_obstacles
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size - 1, self.size - 1]

        # Define static obstacles (can make dynamic later)
        self.obstacles = {(1, 1), (2, 2), (3, 1)}
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        return np.array(self.agent_pos, dtype=np.float32)

    def step(self, action):
        x, y = self.agent_pos
        if action == 0 and y > 0:
            y -= 1  # Up
        elif action == 1 and y < self.size - 1:
            y += 1  # Down
        elif action == 2 and x > 0:
            x -= 1  # Left
        elif action == 3 and x < self.size - 1:
            x += 1  # Right

        # Check for obstacles
        if (x, y) in self.obstacles:
            reward = -10  # Heavy penalty for hitting obstacle
            done = False
        else:
            self.agent_pos = [x, y]
            reward = -1
            done = False

        # Goal check
        if self.agent_pos == self.goal_pos:
            reward = 10
            done = True

        self.steps += 1
        if self.steps >= 100:
            done = True

        return self._get_state(), reward, done

    def render(self, ax=None):
        if ax is None:
            ax = plt.gca()

        ax.clear()
        grid = np.zeros((self.size, self.size))

        for ox, oy in self.obstacles:
            grid[oy, ox] = 0.5  # obstacle gray

        ax.imshow(grid, cmap='gray', vmin=0, vmax=1)

        # Agent is red
        ax.scatter(self.agent_pos[0], self.agent_pos[1], c='red', label='Robot')
        # Goal is green
        ax.scatter(self.goal_pos[0], self.goal_pos[1], c='green', label='Goal')

        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)
        ax.set_title("Grid World")
        ax.legend()
        ax.grid(True)
