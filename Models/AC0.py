import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable
import random
import numpy as np
from collections import deque

# device to run model on
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Using a neural network to learn our policy parameters
class PolicyNetwork(nn.Module):

    # Takes in observations and outputs actions
    def __init__(self, observation_space, action_space, cuda=False):
        super(PolicyNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda == True else "cpu")
        self.input_layer = nn.Linear(observation_space, 128)
        self.hidden_layer = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, action_space)

    # forward pass
    def forward(self, x):
        # input states
        x = self.input_layer(x)
        # relu activation
        x = F.relu(x)
        x = self.hidden_layer(x)
        x = F.relu(x)
        # actions
        actions = self.output_layer(x)

        # get softmax for a probability distribution
        action_probs = F.softmax(actions, dim=1)

        return action_probs


# Using a neural network to learn state value
def masked_softmax(vec, mask, dim=1, epsilon=1e-5):
    exps = torch.exp(vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    return masked_exps / masked_sums


class StateValueNetwork(nn.Module):

    # Takes in state
    def __init__(self, observation_space, cuda=False):
        super(StateValueNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda == True else "cpu")
        self.input_layer = nn.Linear(observation_space, 128)
        self.hidden_layer = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 1)

    def forward(self, x):
        # input layer
        x = self.input_layer(x)
        # activiation relu
        x = F.relu(x)
        x = self.hidden_layer(x)
        x = F.relu(x)
        # get state value
        state_value = self.output_layer(x)

        return state_value


def select_action(network, state, mask):
    ''' Selects an action given current state
    Args:
    - network (Torch NN): network to process state
    - state (Array): Array of action space in an environment

    Return:
    - (int): action that is selected
    - (float): log probability of selecting that action given state and network
    '''

    # convert state to float tensor, add 1 dimension, allocate tensor on device
    state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)

    # use network to predict action probabilities
    action_probs = network(state)
    # state = state.detach()

    mask = Variable(torch.FloatTensor(mask).unsqueeze(0), requires_grad=False).to(DEVICE)

    # sample an action using the probability distribution
    m = Categorical(masked_softmax(action_probs, mask))
    action = m.sample()

    # return action
    return action.item(), m.log_prob(action)


class AC0(nn.Module):
    def __init__(self, inp_dim, action_dim, cuda=True):
        super(AC0, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda == True else "cpu")
        self.epsilon = 0.2
        self.policy = PolicyNetwork(inp_dim, action_dim, cuda)
        self.value = StateValueNetwork(inp_dim)

    def act(self, state, mask):
        bruh = random.random()
        if bruh > self.epsilon:
            # state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            # mask = torch.FloatTensor(mask).unsqueeze(0).to(self.device)
            action, prob = select_action(self.policy, state, mask)
        else:
            indices = np.nonzero(mask)[0]
            randno = random.randint(0, len(indices) - 1)
            action = indices[randno]
            prob = 1.0/len(indices)

        return action, prob

    def act_tensor(self, state, mask):
        bruh = random.random()
        if bruh > self.epsilon:
            action, prob = self.select_action_tensor(state, mask)
        else:
            action = []
            prob = []
            for m in mask:
                indices = np.nonzero(m).squeeze()
                randno = random.randint(0, len(indices) - 1)
                action.append( indices[randno])
                prob.append(1.0/len(indices))
        return torch.tensor(action), torch.tensor(prob)

    def select_action_tensor(self, state, mask):
        """

        :param state: tensor[B, obs] float
        :param mask: tensor[B, obs]  0/1
        :return: action : tensor[B], int
                 m.log_prob(action) tensor[B], float
        """
        action_probs = self.policy(state.float())
        m = Categorical(masked_softmax(action_probs, mask))
        action = m.sample()
        lps = m.log_prob(action)
        return action, lps



    def load_state(self, info):
        self.policy.load_state_dict(info['policy_state_dict'])
        self.value.load_state_dict(info['statevalue_state_dict'])

class Buffer():
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, lp, mask, reward, new_state, new_mask, terminal):
        self.buffer.append((state, action, lp, mask, reward, new_state, new_mask, terminal))

    def sample(self, batch_size):
        states, actions, lps, masks, rewards, new_states, new_mask, terminals = zip(*random.sample(self.buffer, batch_size))
        return states, actions, lps, masks, rewards, new_states, new_mask, terminals
