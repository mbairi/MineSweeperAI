import time
import torch
import torch.nn.functional as F
from Models.AC0 import select_action, AC0, Buffer
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from collections import deque
from game import MineSweeper
from renderer import Render
from numpy import float32
from torch.autograd import Variable
from torch import FloatTensor,LongTensor

#discount factor for future utilities
DISCOUNT_FACTOR = 0.9

BATCH_SIZE = 4096

#device to run model on
device = "cuda" if torch.cuda.is_available() else "cpu"

class Driver():

    def __init__(self, width, height, bomb_no, render_flag):

        self.width = width
        self.height = height
        self.bomb_no = bomb_no
        self.box_count = width * height
        self.render_flag = render_flag
        self.env = MineSweeper(self.width, self.height, self.bomb_no)
        self.envs = [MineSweeper(self.width, self.height, self.bomb_no,rule='win7') for b in range(BATCH_SIZE)]
        self.model = AC0(self.width*self.height, self.width*self.height)
        # Init optimizer
        self.policy_optimizer = optim.SGD(self.model.policy.parameters(), lr=0.001)
        self.stateval_optimizer = optim.SGD(self.model.value.parameters(), lr=0.001)
        self.log = open("./Logs/AC0_log.txt", 'w')

        self.buffer = Buffer(100000)
        self.batch_size = BATCH_SIZE

        if (self.render_flag):
            self.Render = Render(self.env.state)


    def get_action(self, state, mask):
        state = state.flatten()  # vector of len = w * h * 1
        mask = mask.flatten()
        return self.model.act(state, mask)

    def get_action_tensor(self, state, mask):
        """
        Batched version of get action
        :param state: tensor[B, w, h]
        :param mask: tensor[B, w, h]
        :return: tensor[B, action
        """
        state = state.flatten(1)  # tensor[B, w * h]
        mask = mask.flatten(1) # tensor[B, w * h]
        return self.model.act_tensor(state, mask)

    ### Does the action and returns Next State, If terminal, Reward, Next Mask
    def step(self, action):
        i = int(action / self.width)
        j = action % self.width
        if (self.render_flag):
            self.Render.state = self.env.state
            self.Render.draw()
            self.Render.bugfix()
            # time.sleep(0.5)
        next_state, terminal, reward = self.env.choose(i, j)
        next_fog = 1 - self.env.fog
        return next_state, terminal, reward, next_fog

    def step_tensor(self, action):
        """

        :param action: tensor[B]
        :return:
        """
        ii = np.array(action // self.width)
        jj = np.array(action % self.width)  # array [B]
        if self.render_flag:
            self.Render.state = self.envs[0].state
            self.Render.draw()
            self.Render.bugfix()
            time.sleep(0.5)

        next_states, terminals, rewards, next_fogs = [], [], [], []
        for k in range(len(action)):
            next_state, terminal, reward = self.envs[k].choose(ii[k], jj[k])
            next_fog = 1 - self.envs[k].fog
            next_states.append(next_state)
            terminals.append(terminal)
            rewards.append(reward)
            next_fogs.append(next_fog)

        next_states, terminals, rewards, next_fogs = torch.tensor(next_states), \
            torch.tensor(terminals), torch.tensor(rewards), torch.tensor(next_fogs)
        return next_states, terminals, rewards, next_fogs

    def Loss(self, state, lp, reward, next_state, terminal):


        state_val = self.model.value(state.flatten(1).float())
        new_state_val = self.model.value(next_state.flatten(1).float())

        # calculate value function loss with MSE
        val_loss = F.mse_loss(reward + DISCOUNT_FACTOR * new_state_val * (1-terminal.int()), state_val)
        val_loss *= DISCOUNT_FACTOR

        # calculate policy loss
        advantage = torch.mean(- lp * (reward + (DISCOUNT_FACTOR * new_state_val - state_val)*(1-terminal.int())))
        policy_loss = advantage * DISCOUNT_FACTOR

        # Backpropagate policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph = True)
        self.policy_optimizer.step()

        # Backpropagate value
        self.stateval_optimizer.zero_grad()
        val_loss.backward()
        self.stateval_optimizer.step()

        return policy_loss.item(), val_loss.item()

    def save_checkpoints(self, batch_no):
        # path = "./pre-trained/AC0" + str(batch_no) + ".pth"
        path = "./pre-trained/ac0_dnn" + str(batch_no) + ".pth"
        torch.save({
            'epoch': batch_no,
            'policy_state_dict': self.model.policy.state_dict(),
            'statevalue_state_dict':self.model.value.state_dict()
        }, path)

    def save_logs(self, batch_no, avg_reward, policy_loss, val_loss,  wins):
        res = [
            str(batch_no),
            "\tAvg Reward: ",
            str(avg_reward),
            "\t Policy Loss: ",
            str(policy_loss),
            "\t Val Loss: ",
            str(val_loss),
            "\t Wins: ",
            str(wins),
        ]
        log_line = " ".join(res)
        print(log_line)
        self.log.write(log_line + "\n")
        self.log.flush()

def main():
    driver = Driver(6, 6, 6, False)
    save_every = 2000
    epochs = 100000
    wins = [deque(maxlen=100) for b in range(BATCH_SIZE)]
    for t in range(BATCH_SIZE):
        wins[t].append(0)

    for epoch in range(epochs):

        mask = torch.tensor([1 - driver.envs[k].fog for k in range(BATCH_SIZE)])
        state = torch.tensor([driver.envs[k].state for k in range(BATCH_SIZE)])
        action, lp = driver.get_action_tensor(state, mask)
        next_state, terminal, reward, _ = driver.step_tensor(action)

        for done_id in range(BATCH_SIZE) :
            if terminal[done_id]:
                driver.envs[done_id].reset()

        policy_loss, val_loss = driver.Loss(state, lp, reward, next_state, terminal)

        for t in range(BATCH_SIZE):
            if reward[t].item() > 0.8:
                wins[t].append(1)
            elif reward[t].item() < -0.8:
                wins[t].append(0)

        driver.save_logs(epoch, reward.float().mean().item(), policy_loss, val_loss, np.mean([np.mean(win) for win in wins]))

        # Saves the model details to "./pre-trained" if 1000 batches have been processed
        if (epoch % save_every == 0):
            driver.save_checkpoints(epoch)


if __name__ == "__main__":
    main()
