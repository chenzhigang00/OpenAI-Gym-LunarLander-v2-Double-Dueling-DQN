import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from experience_replay import ReplayBuffer
from qnet import QNetwork

LR = 5e-4                
BUFFER_SIZE = int(1e5)   
BATCH_SIZE = 64          
GAMMA = 0.99             
TAU = 1e-3               # 目标网络参数软更新

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size, action_size): 
        self.state_size = state_size
        self.action_size = action_size
        # DDQN: 建模两个独立的 Q函数神经网络，在动作神经网络上计算最大动作，但在目标神经网络上取值，目的是减小 Q函数过估计问题
        # 原来 DQN是动作选择及取值都在目标网络上，造成了 Q值过估计
        self.qnetwork_local = QNetwork(state_size, action_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)   # 经验重放池
        
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)        
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()   # 采样
            self.learn(experiences, GAMMA)  # 更新
                
    def act(self, state, eps):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        self.qnetwork_local.eval()
        with torch.no_grad():
            # DDQN
            Q_pred = self.qnetwork_local(next_states)   # 在动作网络上计算最大动作
            max_actions = torch.argmax(Q_pred, dim=1).long().unsqueeze(1)    
            Q_next = self.qnetwork_target(next_states)  # 在目标网络上取值
        self.qnetwork_local.train()
        Q_targets = rewards + (gamma * Q_next.gather(1, max_actions) * (1.0 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)        

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)   # 对目标网络软更新
        
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)   # 目标网络参数产生变化，而非直接替代
        