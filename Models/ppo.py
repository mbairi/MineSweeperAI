import torch
import torch.nn as nn
from torch.autograd import Variable
from torchsummary import summary
from collections import deque
import numpy as np
import random
from torch.distributions.categorical import Categorical

class Actor(nn.Module):
    
    def __init__(self, inp_dim, action_dim):
        super(Actor, self).__init__()
        
        self.epsilon = 1
        self.actor = nn.Sequential(
            nn.Linear(inp_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128,action_dim)
        )        

    
    ### This is important, masks invalid actions
    def masked_softmax(self,vec, mask, dim=1, epsilon=1e-5):
            exps = torch.exp(vec)
            masked_exps = exps * mask.float()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return (masked_exps/masked_sums)
        
    def forward(self, x,mask):
        x=x/8
        x = self.feature(x)
        dist = self.masked_softmax(self.actor(x),mask)
        dist = Categorical(dist)
        return dist
    
    # def act(self,state,mask):
    #     bruh = random.random()
    #     if bruh > self.epsilon:
    #         state   = torch.FloatTensor(state).unsqueeze(0)
    #         mask   = torch.FloatTensor(mask).unsqueeze(0)
    #         q_value = self.forward(state,mask)
    #         # print(q_value)
    #         action  = q_value.max(1)[1].data[0].item()
    #     else:
    #         indices = np.nonzero(mask)[0]
    #         randno = random.randint(0,len(indices)-1)
    #         action = indices[randno]
    #     return action


class Critic(nn.Module):
    
    def __init__(self, inp_dim):
        super(Critic, self).__init__()
        
        self.epsilon = 1
        self.critic = nn.Sequential(
            nn.Linear(inp_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128,1)
        )        

    
    ### This is important, masks invalid actions

        
    def forward(self, x,mask):
        x=x/8
        x = self.feature(x)
        dist = self.actor(x)
        return dist
    
    # def act(self,state,mask):
    #     bruh = random.random()
    #     if bruh > self.epsilon:
    #         state   = torch.FloatTensor(state).unsqueeze(0)
    #         mask   = torch.FloatTensor(mask).unsqueeze(0)
    #         q_value = self.forward(state,mask)
    #         # print(q_value)
    #         action  = q_value.max(1)[1].data[0].item()
    #     else:
    #         indices = np.nonzero(mask)[0]
    #         randno = random.randint(0,len(indices)-1)
    #         action = indices[randno]
    #     return action


class Buffer():
    def __init__(self,capacity):
        self.buffer = deque(maxlen = capacity)

    def push(self,state,probs,val,action,mask,reward,terminal):
        self.buffer.append((state,probs,val,action,mask,reward,terminal))

    def sample(self,batch_size):
        states,probs,vals,actions,masks,rewards,terminals = zip(*random.sample(self.buffer, batch_size))
        return states,probs,vals,actions,masks,rewards,terminals
