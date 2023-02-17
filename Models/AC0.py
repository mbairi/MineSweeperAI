import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable
import random


#discount factor for future utilities
DISCOUNT_FACTOR = 0.99

#number of episodes to run
NUM_EPISODES = 1000

#max steps per episode
MAX_STEPS = 10000

#score agent needs for environment to be solved
SOLVED_SCORE = 195

#device to run model on
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Using a neural network to learn our policy parameters
class PolicyNetwork(nn.Module):

    # Takes in observations and outputs actions
    def __init__(self, observation_space, action_space,cuda=True):
        super(PolicyNetwork, self).__init__()
        self.device=torch.device("cuda" if torch.cuda.is_available() and cuda==True else "cpu")      
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
    def __init__(self, observation_space):
        super(StateValueNetwork, self).__init__()

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
    def __init__(self, inp_dim, action_dim,cuda=True):
        super(AC0, self).__init__()
        self.device=torch.device("cuda" if torch.cuda.is_available() and cuda==True else "cpu")
        self.epsilon = 1
        self.network=PolicyNetwork(inp_dim, action_dim, cuda)
        

    def act(self,state,mask):
        bruh = random.random()
        if bruh > self.epsilon:
            state   = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            mask   = torch.FloatTensor(mask).unsqueeze(0).to(self.device)
            action_probs = self.network(state)
            m = Categorical(masked_softmax(action_probs, mask))
            action = m.sample()
        else:
            indices = np.nonzero(mask)[0]
            randno = random.randint(0,len(indices)-1)
            action = indices[randno]
        action=int(action)
        return action

    def load_state(self,info):
        self.network.load_state_dict(info['policy_state_dict'])
