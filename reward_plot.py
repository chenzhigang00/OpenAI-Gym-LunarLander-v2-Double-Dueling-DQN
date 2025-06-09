import numpy as np
import matplotlib.pyplot as plt

rewards = np.load("OpenAI-Gym-LunarLander-v2-Double-Dueling-DQN/rewards_3.npy")
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Reward Curve")
plt.grid(True)
plt.show()
