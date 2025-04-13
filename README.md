# Robot_navigation_DRL_Pytorch

This project implements a Deep Reinforcement Learning (DRL) approach for training a virtual robot to navigate a 2D grid world environment with obstacles and reach a specified goal. The robot uses the Deep Q-Network (DQN) algorithm to learn optimal actions through trial and error over multiple episodes.

Key Components:

Environment (GridWorld): A customizable grid map where the robot starts from a random position, avoids obstacles (black squares), and aims to reach a goal (green square).
Agent (DQN): A neural network that approximates Q-values and learns the best actions based on state input.
Replay Buffer: Stores experience tuples (state, action, reward, next_state, done) to enable experience replay and improve sample efficiency.
Training (train.py): Trains the DQN agent using the environment and saves the best-performing model.
Visualization (visualize.py): Loads the trained model and animates the robot's movements on the grid as it navigates toward the goal in a single, interactive window.
