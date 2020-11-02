import torch
import torch.nn as nn
from torch.autograd import Variable
from torchsummary import summary
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self,inp_dim,hidden_dim):
        super().__init__()

        self.inp_dim = inp_dim
        self.hidden_dim = hidden_dim
        self.epsilon = 0.999

        self.model = nn.Sequential(
            nn.Linear(inp_dim,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,inp_dim),
            nn.Softmax(dim=-1)
        )
        

    def forward(self,x,mask):
        x=self.model(x)
    
    def act(self,state):
        bruh = random.random()
        if bruh > self.epsilon:
            state   = Variable(torch.FloatTensor(state).unsqueeze(0), requires_grad=False)
            q_value = self.model(state)
            action  = q_value.max(1)[1].data[0].item()
        else:
            action = random.randrange(self.inp_dim)
        return action

class CNNDQN(nn.Module):
    def __init__(self,inp_dim):
        super().__init__()

        self.inp_dim = inp_dim
        self.epsilon = 0.999

        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0),
        )

        self.fc = nn.Sequential(
            nn.Linear(75,inp_dim*inp_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self,x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def act(self,state):
        bruh = random.random()
        if bruh > self.epsilon:
            state   = Variable(torch.FloatTensor(state).unsqueeze(0), requires_grad=False).unsqueeze(1)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0].item()
        else:
            action = random.randrange(self.inp_dim*self.inp_dim)
        return action


class Buffer():
    def __init__(self,capacity):
        self.buffer = deque(maxlen = capacity)

    def push(self,state,action,reward,new_state,terminal):
        self.buffer.append((state,action,reward,new_state,terminal))

    def sample(self,batch_size):
        states,actions,rewards,new_states,terminals = zip(*random.sample(self.buffer, batch_size))
        return states,actions,rewards,new_states,terminals



