# Import the Necessary Packages
import gymnasium as gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from doubleDQN_agent import Agent
from qnet import QNetwork

def seed_everything(seed=39):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

env = gym.make("LunarLander-v3", continuous = False, gravity = -10.0,
               enable_wind = False, wind_power=15.0, turbulence_power=1.5, render_mode=None)  # LunarLander-v3 with Gymnasium
seed_everything()
env.reset(seed=39)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = Agent(state_size=state_dim, action_size=action_dim)

n_episodes = 1500
max_t=1000
eps_start=1.0
eps_end=0.01
eps_decay=0.995
episode_rewards = []
eps = eps_start

for i_episode in range(1, n_episodes+1):
    state, _ = env.reset()
    score = 0
    for t in range(max_t):      # 采样
        action = agent.act(state, eps)
        next_state, reward, terminated, truncated, _= env.step(action)
        done = terminated or truncated
        agent.step(state, action, reward, next_state, done)
        score += reward
        state = next_state
        if done:
            break
    episode_rewards.append(score)
    if i_episode % 10 == 0:
        avg = np.mean(episode_rewards[-10:])
        print(f"Episode {i_episode}, Avg Reward: {avg:.2f}")
    eps = max(eps_end, eps_decay*eps)
    

torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_3.pth')

np.save("rewards_3.npy", episode_rewards)
