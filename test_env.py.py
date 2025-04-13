from gridworld import GridWorld

env = GridWorld()
state = env.reset()
print("Start state:", state)

actions = [1, 1, 3, 3, 1, 3]  # down, down, right, right, down, right
for action in actions:
    new_state, reward, done = env.step(action)
    print(f"Action: {action}, State: {new_state}, Reward: {reward}, Done: {done}")
    if done:
        break
