import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import gym
import cv2

import time

from ultralytics import YOLO
from GameENV import ENV

bbox_detector_path = './The_King_of_Fighters_XV/model/bbox_detector.pt'
Action_detector_path = './The_King_of_Fighters_XV/model/action_detector.pt'

# DQN网络
class DQN(nn.Module):
    def __init__(self, output_dim):
        super(DQN, self).__init__()
        self.mobilev3 = models.mobilenet_v3_small(pretrained=True)
        self.mobilev3.classifier = nn.Linear(576, output_dim)

    def forward(self, x):
        x = self.mobilev3.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.mobilev3.classifier(x)
        return x

# 经验回放
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = [torch.tensor(x).squeeze().cuda().item() if torch.is_tensor(x) else x for x in zip(*batch)]
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, env):
        self.env = env

        self.input_dim = 15   # 获取状态空间维度
        self.output_dim = env.action_space.n # 获取行为空间维度

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 检查是否有GPU

        self.policy_net = DQN(self.output_dim).to(self.device)   # 创建策略网络
        self.target_net = DQN(self.output_dim).to(self.device)   # 创建目标网络
        self.target_net.load_state_dict(self.policy_net.state_dict())   # 将策略网络的参数复制到目标网络中
        self.target_net.eval()   # 将目标网络设置为评估模式
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)   # 创建Adam优化器

        self.gamma = 0.99   # 奖励折扣因子
        self.epsilon = 1.0   # ε-greedy策略的初始探索率
        self.epsilon_min = 0.01   # ε-greedy策略的最小探索率
        self.epsilon_decay = 0.995   # ε-greedy策略的探索率衰减因子
        self.batch_size = 16   # 批量大小
        self.memory = ReplayBuffer(10000)   # 创建经验回放缓冲区

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.select_randomAction()
        
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state.view(-1,3,self.input_dim//3, 1))
            action = q_values.max(1)[1].item()
        return action

    def train(self):
        if len(self.memory) < self.batch_size:
            # 如果小于batchsize，跳过训练
            return

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        # 计算当前状态-操作对和下一个状态的 Q 值
        q_values = self.policy_net(state.view(-1,3,self.input_dim//3,1)).gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_state.view(-1,3,self.input_dim//3,1)).max(1)[0]

        # 使用贝尔曼方程计算预期的 Q 值
        expected_q_values = reward + self.gamma * next_q_values * (1 - done)

        # 计算预测 Q 值与预期 Q 值之间的损耗
        loss = nn.MSELoss()(q_values, expected_q_values.detach())

        print("train start")
        # 使用反向传播更新策略网络的权重
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def run(self, episodes):
        for episode in range(episodes):
            state = self.env.get_state()
            done = False
            total_reward = 0
            while not done:
                # Select an action using the epsilon-greedy strategy
                action = self.select_action(state)
                print(f"action: {action}")
                # Take a step in the environment with the selected action
                next_state, reward, done = self.env.step(action)
                
                # Store the experience in the replay buffer
                self.memory.push(state, action, reward, next_state, done)

                # Train the network using a batch of experiences from the replay buffer
                self.train()
                print("train done")
                # Update the current state and total reward
                state = next_state
                total_reward += reward

            self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
            self.update_target_model()
            print('Episode: {}/{}, Total reward: {}, Epsilon: {:.2f}'
                  .format(episode+ 1, episodes, total_reward, self.epsilon))


if __name__ == "__main__":
    Box_detector = YOLO(bbox_detector_path)
    Action_detector = YOLO(Action_detector_path)

    env = ENV(
        window_size=(0,0,990,560),
        bbox_detector = Box_detector,
        action_detector = Action_detector,
    )

    dqn_agent = DQNAgent(env)

    dqn_agent.run(episodes=2)