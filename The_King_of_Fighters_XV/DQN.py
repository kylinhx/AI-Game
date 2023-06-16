import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import gym
import cv2

# Define the DQN network
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

# Define Experience Replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        # Add new experience to the buffer
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # Sample a batch from the buffer
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, env):
        # Initialize agent with environment
        self.env = env
        self.input_dim = env.observation_space.shape[0]   # 获取状态空间维度
        self.output_dim = env.action_space.n   # 获取行为空间维度
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 检查是否有GPU

        # Define DQN networks and optimizer
        self.policy_net = DQN(self.output_dim).to(self.device)   # 创建策略网络
        self.target_net = DQN(self.output_dim).to(self.device)   # 创建目标网络
        self.target_net.load_state_dict(self.policy_net.state_dict())   # 将策略网络的参数复制到目标网络中
        self.target_net.eval()   # 将目标网络设置为评估模式
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)   # 创建Adam优化器

        # Define hyperparameters
        self.gamma = 0.99   # 奖励折扣因子
        self.epsilon = 1.0   # ε-greedy策略的初始探索率
        self.epsilon_min = 0.01   # ε-greedy策略的最小探索率
        self.epsilon_decay = 0.995   # ε-greedy策略的探索率衰减因子
        self.batch_size = 128   # 批量大小
        self.memory = ReplayBuffer(10000)   # 创建经验回放缓冲区

    def select_action(self, state):
        # Choose an action using epsilon-greedy
        if np.random.rand() < self.epsilon:
            # Select a random action with probability ε
            return self.env.action_space.sample()
        with torch.no_grad():
            # Select the action with the highest Q-value with probability (1-ε)
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            action = q_values.max(1)[1].item()
        return action

    def train(self):
        # Train the network using sampled experiences
        if len(self.memory) < self.batch_size:
            # If the replay buffer does not have enough experiences, skip training
            return

        # Sample a batch of experiences from the replay buffer
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        # Compute the Q-values for the current state-action pairs and the next states
        q_values = self.policy_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_state).max(1)[0]

        # Compute the expected Q-values using the Bellman equation
        expected_q_values = reward + self.gamma * next_q_values * (1 - done)

        # Compute the loss between the predicted Q-values and the expected Q-values
        loss = nn.MSELoss()(q_values, expected_q_values.detach())

        # Update the weights of the policy network using backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        # Update the target network with the policy network's weights
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def run(self, episodes):
        # Run the agent for a specified number of episodes
        for episode in range(episodes):
            # Reset the environment for a new episode
            state = self.env.reset()[0] # sometimes returned state will be ([state_dim1, state_dim2...], {})
            done = False
            total_reward = 0
            while not done:
                # Select an action using the epsilon-greedy strategy
                action = self.select_action(state)

                # Take a step in the environment with the selected action
                next_state, reward, done, _, _ = self.env.step(action)

                # Store the experience in the replay buffer
                self.memory.push(state, action, reward, next_state, done)

                # Train the network using a batch of experiences from the replay buffer
                self.train()

                # Update the current state and total reward
                state = next_state
                total_reward += reward

            # Update the ε-greedy exploration rate
            self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

            # Update the target network with the weights of the policy network
            self.update_target_model()

            # Print the episode number, the total reward, and the current ε-greedy exploration rate
            print('Episode: {}/{}, Total reward: {}, Epsilon: {:.2f}'
                  .format(episode+ 1, episodes, total_reward, self.epsilon))

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Create the DQN agent and run it
dqn_agent = DQNAgent(env)
dqn_agent.run(episodes=200)