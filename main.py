# Import the Necessary Packages
import gymnasium as gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from doubleDQN_agent import Agent
from qnet import QNetwork

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

env = gym.make("LunarLander-v3", continuous = False, gravity = -10.0,
               enable_wind = False, wind_power=15.0, turbulence_power=1.5, render_mode=None)  # LunarLander-v3 with Gymnasium
seed_everything()
env.reset(seed=42)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = Agent(state_size=state_dim, action_size=action_dim)

agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

for i in range(5):
    state = env.reset()
    img = plt.imshow(env.render(mode='rgb_array'))
    for j in range(600):
        action = agent.act(state)
        img.set_data(env.render(mode='rgb_array')) 
        plt.axis('off')
        state, reward, done, _ = env.step(action)
        if done:
            break 
            
env.close()