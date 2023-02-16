import time
import torch
import torch.nn.functional as F
from Models.AC0 import PolicyNetwork, StateValueNetwork, select_action
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from collections import deque
from game import MineSweeper
from renderer import Render

#discount factor for future utilities
DISCOUNT_FACTOR = 0.99

#number of episodes to run
NUM_EPISODES = 10000000

#max steps per episode
MAX_STEPS = 50

#score agent needs for environment to be solved
SOLVED_SCORE = 0.8

#device to run model on
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Driver():

    def __init__(self, width, height, bomb_no, render_flag):

        self.width = width
        self.height = height
        self.bomb_no = bomb_no
        self.box_count = width * height
        self.render_flag = render_flag
        self.env = MineSweeper(self.width, self.height, self.bomb_no)
        self.policy_network = PolicyNetwork(width*height, width*height).to(DEVICE)
        self.stateval_network = StateValueNetwork(width*height).to(DEVICE)
        # Init optimizer
        self.policy_optimizer = optim.SGD(self.policy_network.parameters(), lr=0.01)
        self.stateval_optimizer = optim.SGD(self.stateval_network.parameters(), lr=0.01)
        self.log = open("./Logs/AC0_log.txt", 'w')

        # track scores
        self.scores = []
        # track recent scores
        self.recent_scores = deque(maxlen=100)

        if (self.render_flag):
            self.Render = Render(self.env.state)


    def get_action(self, network, state, mask):
        state = state.flatten()  # vector of len = w * h * 1
        mask = mask.flatten()
        return select_action(network, state, mask)

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

    def save_checkpoints(self, batch_no):
        path = "./pre-trained/AC0" + str(batch_no) + ".pth"
        torch.save({
            'epoch': batch_no,
            'policy_state_dict': self.policy_network.state_dict(),
            'statevalue_state_dict':self.stateval_network.state_dict()
        }, path)

    def save_logs(self, batch_no, avg_reward, wins_mean):
        res = [
            str(batch_no),
            "\tAvg Reward: ",
            str(avg_reward),
            "Avg wins: ",
            str(wins_mean),
        ]
        log_line = " ".join(res)
        if wins_mean > 0.75:
            print(log_line)
        self.log.write(log_line + "\n")
        self.log.flush()

def main():
    driver = Driver(6, 6, 3, False)
    save_every = 50000
    wins = deque([0]*100)
    for episode in tqdm(range(NUM_EPISODES)):
        if np.mean(wins) > 0.9:
            print(0.9)
            break

        driver.env.reset()
        # init variables
        state = driver.env.state
        done = False
        score = 0
        win = 0
        I = 1

        # run episode, update online
        for step in range(MAX_STEPS):

            mask = 1 - driver.env.fog
            # get action and log probability
            action, lp = driver.get_action(driver.policy_network, state, mask)

            # step with action
            new_state, done, reward, next_fog = driver.step(action)

            # update episode score
            score += reward

            # get state value of current state
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
            state_val = driver.stateval_network(state_tensor.flatten())

            # get state value of next state
            new_state_tensor = torch.from_numpy(new_state).float().unsqueeze(0).to(DEVICE)
            new_state_val = driver.stateval_network(new_state_tensor.flatten())

            # if terminal state, next state val is 0
            if done:
                new_state_val = torch.tensor([0]).float().unsqueeze(0).to(DEVICE)

            # calculate value function loss with MSE
            val_loss = F.mse_loss(reward + DISCOUNT_FACTOR * new_state_val, state_val)
            val_loss *= I

            # calculate policy loss
            advantage = reward + DISCOUNT_FACTOR * new_state_val.item() - state_val.item()
            policy_loss = -lp * advantage
            policy_loss *= I

            # Backpropagate policy
            driver.policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            driver.policy_optimizer.step()

            # Backpropagate value
            driver.stateval_optimizer.zero_grad()
            val_loss.backward()
            driver.stateval_optimizer.step()

            if done:
                if (reward == 1):
                    win += 1
                break

            # move into new state, discount I
            state = new_state
            I *= DISCOUNT_FACTOR

        wins.append(win)
        wins.popleft()

        # append episode score
        driver.scores.append(score)
        driver.recent_scores.append(score)

        if episode % 100 == 0:
            driver.save_logs(episode, np.mean(driver.recent_scores), np.mean(wins))

        if episode % save_every == 0:
            driver.save_checkpoints(episode)



if __name__ == "__main__":
    main()
