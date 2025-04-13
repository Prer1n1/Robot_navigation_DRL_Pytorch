import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gridworld import GridWorld
from dqn_agent import DQN
import time

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = DQN().to(device)
model.load_state_dict(torch.load("dqn_robot.pth", map_location=device))
model.eval()

# Create the environment
env = GridWorld()
state = env.reset()
done = False
grid_size = env.size

# Set up the plot (only once)
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Robot Navigation with Obstacles")

# Draw grid lines
for x in range(grid_size + 1):
    ax.axhline(x, lw=1, color='gray')
    ax.axvline(x, lw=1, color='gray')

# Draw obstacles
for obs in env.obstacles:
    patch = patches.Rectangle(obs, 1, 1, color='black')
    ax.add_patch(patch)

# Draw goal
goal_patch = patches.Rectangle(env.goal, 1, 1, color='green', alpha=0.5)
ax.add_patch(goal_patch)

# Draw robot
robot_patch = patches.Circle((state[0] + 0.5, state[1] + 0.5), 0.3, color='blue')
ax.add_patch(robot_patch)

# Enable interactive mode
plt.ion()
plt.show()

# Start robot animation
while not done:
    # Update robot position
    robot_patch.center = (state[0] + 0.5, state[1] + 0.5)
    fig.canvas.draw()
    plt.pause(0.3)

    # Choose action using trained model
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        state_tensor = state_tensor / (env.size - 1)  # Normalize input
        q_values = model(state_tensor)
        action = torch.argmax(q_values).item()

    # Perform action
    next_state, reward, done = env.step(action)
    state = next_state

# Final update after reaching goal
robot_patch.center = (state[0] + 0.5, state[1] + 0.5)
fig.canvas.draw()
plt.pause(1.0)
plt.ioff()
plt.show()
