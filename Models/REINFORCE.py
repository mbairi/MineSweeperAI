
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):

    # Takes in observations and outputs actions
    def __init__(self, observation_space, action_space, cuda=False):
        super(PolicyNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda == True else "cpu")
        self.input_layer = nn.Linear(observation_space, 128)
        self.output_layer = nn.Linear(128, action_space)

    # forward pass
    def forward(self, x):
        # input states
        x = self.input_layer(x)
        # relu activation
        x = F.tanh(x)
        # actions
        actions = self.output_layer(x)
        #print("action:", actions)

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
        x = F.tanh(x)
        x = self.hidden_layer(x)
        x = F.tanh(x)
        # get state value
        state_value = self.output_layer(x)

        return state_value
