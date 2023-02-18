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

#max steps per episode
MAX_STEPS = 50

#score agent needs for environment to be solved
SOLVED_SCORE = 0.8

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
        self.model = AC0(self.width*self.height, self.width*self.height)
        # Init optimizer
        self.policy_optimizer = optim.SGD(self.model.policy.parameters(), lr=0.001)
        self.stateval_optimizer = optim.SGD(self.model.value.parameters(), lr=0.001)
        self.log = open("./Logs/AC0_log.txt", 'w')

        self.buffer = Buffer(100000)
        self.batch_size = 4096

        if (self.render_flag):
            self.Render = Render(self.env.state)


    def get_action(self, state, mask):
        state = state.flatten()  # vector of len = w * h * 1
        mask = mask.flatten()
        return self.model.act(state, mask)

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

    def Loss(self):

        self.model.train()
        ### Samples batch from buffer memory
        state, action, lp, mask, reward, next_state, next_mask, terminal = self.buffer.sample(self.batch_size)
        ### Converts the variabls to tensors for processing by DDQN
        state = FloatTensor(float32(state)).to(device)
        next_state = FloatTensor(float32(next_state)).to(device)
        reward = FloatTensor(reward).unsqueeze(1).to(device)
        lp = FloatTensor(lp).unsqueeze(1).to(device)


        # get state value of current state
        # state_tensor = state.float().unsqueeze(0).to(device)
        state_val = self.model.value(state)

        # get state value of next state
        # next_state_tensor = next_state.float().unsqueeze(0).to(device)
        new_state_val = self.model.value(next_state)

        # calculate value function loss with MSE
        val_loss = F.mse_loss(reward + DISCOUNT_FACTOR * new_state_val, state_val)
        val_loss *= DISCOUNT_FACTOR

        # calculate policy loss
        advantage = torch.mean(- lp * (reward + DISCOUNT_FACTOR * new_state_val - state_val))
        policy_loss = advantage * DISCOUNT_FACTOR

        # Backpropagate policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        # Backpropagate value
        self.stateval_optimizer.zero_grad()
        val_loss.backward()
        self.stateval_optimizer.step()

        self.model.eval()

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
    driver = Driver(9, 9, 10, False)
    state = driver.env.state
    save_every = 2000
    epochs = 10000
    batch_no = 0
    count = 0
    wins = 0
    running_reward = 0
    total = 0

    while (batch_no < epochs):

        # simple state action reward loop and writes the actions to buffer
        mask = 1 - driver.env.fog
        action, lp = driver.get_action(state, mask)
        next_state, terminal, reward, _ = driver.step(action)
        driver.buffer.push(state.flatten(), action, lp, mask.flatten(), reward, next_state.flatten(),
                           (1 - driver.env.fog).flatten(), terminal)
        state = next_state
        count += 1
        running_reward += reward

        # Used for calculating winrate for each batch
        if (terminal):
            if (reward == 1):
                wins += 1
            driver.env.reset()
            state = driver.env.state
            mask = driver.env.fog
            total += 1

        if count == driver.batch_size:

            policy_loss, val_loss = driver.Loss()

            # Calculates metrics
            batch_no += 1
            avg_reward = running_reward / driver.batch_size
            wins = wins * 100 / total
            driver.save_logs(batch_no, avg_reward, policy_loss, val_loss, wins)

            # Resets metrics for next batch calculation
            running_reward = 0
            count = 0
            wins = 0
            total = 0

            # Saves the model details to "./pre-trained" if 1000 batches have been processed
            if (batch_no % save_every == 0):
                driver.save_checkpoints(batch_no)


if __name__ == "__main__":
    main()
