import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from gridworld import GridWorld
from dqn_agent import DQN, ReplayBuffer
import matplotlib.pyplot as plt

# Set seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
EPISODES = 1000
BATCH_SIZE = 32
GAMMA = 0.99
LR = 0.001
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995

# Initialize environment, model, target model, optimizer, and replay buffer
env = GridWorld()
model = DQN().to(device)
target_model = DQN().to(device)
target_model.load_state_dict(model.state_dict())  # Initially copy weights

optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()
memory = ReplayBuffer()

# Exploration factor
epsilon = EPSILON_START

# Tracking rewards
rewards_per_episode = []
best_reward_per_episode = []
best_reward = float('-inf')  # Set best reward to lowest possible initially

# === TRAINING LOOP ===
for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0

    for step in range(100):  # Max steps per episode
        # Normalize state to [0, 1]
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device) / (env.size - 1)

        # Epsilon-greedy policy
        if random.random() < epsilon:
            action = random.randint(0, 3)  # Random action (explore)
        else:
            with torch.no_grad():
                q_values = model(state_tensor)
                action = torch.argmax(q_values).item()  # Best action (exploit)

        # Take the action
        next_state, reward, done = env.step(action)
        total_reward += reward

        # Store experience
        memory.push((state, action, reward, next_state, done))
        state = next_state

        # Start learning only after enough experiences
        if len(memory) >= 1000:
            batch = memory.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            # Convert to tensors
            states = torch.tensor(states, dtype=torch.float32).to(device) / (env.size - 1)
            next_states = torch.tensor(next_states, dtype=torch.float32).to(device) / (env.size - 1)
            actions = torch.tensor(actions).unsqueeze(1).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
            dones = torch.tensor(dones, dtype=torch.bool).unsqueeze(1).to(device)

            # Current Q values
            q_values = model(states).gather(1, actions)

            # Target Q values
            with torch.no_grad():
                next_q_values = target_model(next_states).max(1)[0].unsqueeze(1)
                target_q = rewards + GAMMA * next_q_values * (~dones)

            # Compute loss and update model
            loss = loss_fn(q_values, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    # Decay epsilon
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    # Save reward
    rewards_per_episode.append(total_reward)
    best_reward_per_episode.append(best_reward)

    # Save best model if improved
    if total_reward > best_reward:
        best_reward = total_reward
        torch.save(model.state_dict(), "best_dqn_robot.pth")
        print(f"🚀 New best model saved at episode {episode+1} with reward: {best_reward:.2f}")

    # Update target network every 10 episodes
    if episode % 10 == 0:
        target_model.load_state_dict(model.state_dict())

    # Print training progress
    print(f"Episode {episode+1}/{EPISODES} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f} | Memory Size: {len(memory)}")

# Save final model
torch.save(model.state_dict(), "dqn_robot.pth")
print("\n Final model saved as 'dqn_robot.pth'")

# === PLOTTING RESULTS ===
plt.figure(figsize=(12, 6))
plt.plot(rewards_per_episode, label="Reward per Episode", color="blue")
plt.plot(best_reward_per_episode, label="Best Reward So Far", linestyle="--", color="orange")
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN Robot Learning Progress')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
