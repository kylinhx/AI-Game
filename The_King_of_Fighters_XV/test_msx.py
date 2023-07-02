import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import gym
from gym import envs
import cv2

env = gym.make('CartPole-v1')
print(env.reset())