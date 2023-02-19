from game import MineSweeper
import torch
import torch.optim as optim
import torch.nn as nn
from Models.REINFORCE import PolicyNetwork, StateValueNetwork, masked_softmax
import numpy as np
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = MineSweeper(9, 9, 10)
action_size = 81
state_size = 81

# hyperparameters
episodes = 5000  # run agent for this many episodes
actor_lr = 0.002  # learning rate for actor
value_function_lr = 0.002  # learning rate for value function
discount = 0.99  # discount factor gamma value
reward_scale = 0.01  # scale reward by this amount

stats_rewards_list = []  # store stats for plotting in this
stats_every = 10  # print stats every this many episodes
total_reward = 0
timesteps = 0
episode_length = 0
stats_actor_loss, stats_vf_loss = 0., 0.


class PGAgent():
    def __init__(self, state_size, action_size, actor_lr, vf_lr, discount):
        self.action_size = action_size
        self.actor_net = PolicyNetwork(state_size, action_size).to(device)
        self.vf_net = StateValueNetwork(state_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.vf_optimizer = optim.Adam(self.vf_net.parameters(), lr=vf_lr)
        self.discount = discount

    def select_action(self, state, mask):
        # get action probs then randomly sample from the probabilities
        with torch.no_grad():
            input_state = torch.FloatTensor(state.flatten()).unsqueeze(0).to(device)
            output = self.actor_net(input_state) # explode
            action_probs = masked_softmax(output, torch.FloatTensor(mask.flatten()).unsqueeze(0))
            # detach and turn to numpy to use with np.random.choice()
            m = Categorical(action_probs)
            action = m.sample()

        return action.item()

    def save_checkpoints(self, batch_no):
        # path = "./pre-trained/AC0" + str(batch_no) + ".pth"
        path = "./pre-trained/reinforce_dnn" + str(batch_no) + ".pth"
        torch.save({
            'epoch': batch_no,
            'policy_state_dict': self.actor_net.state_dict(),
            'statevalue_state_dict':self.vf_net.state_dict()
        }, path)

    def train(self, state_list, mask_list, action_list, reward_list):
        # turn rewards into return
        trajectory_len = len(reward_list)
        return_array = np.zeros((trajectory_len,))
        g_return = 0.
        for i in range(trajectory_len - 1, -1, -1):
            g_return = reward_list[i] + self.discount * g_return
            return_array[i] = g_return

        # create tensors
        state_t = torch.FloatTensor(state_list).flatten(1).to(device)
        mask_t = torch.FloatTensor(mask_list).flatten(1).to(device)
        action_t = torch.LongTensor(action_list).to(device).view(-1, 1)
        return_t = torch.FloatTensor(return_array).to(device).view(-1, 1)

        # get value function estimates
        vf_t = self.vf_net(state_t).to(device)
        with torch.no_grad():
            advantage_t = return_t - vf_t

        # calculate actor loss
        selected_action_prob = self.actor_net(state_t).gather(1, action_t)
        # REINFORCE loss:
        # actor_loss = torch.mean(-torch.log(selected_action_prob) * return_t)
        # REINFORCE Baseline loss:
        actor_weight = self.actor_net.state_dict()
        actor_loss = torch.mean(-torch.log(selected_action_prob) * advantage_t)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # calculate vf loss
        loss_fn = nn.MSELoss()
        vf_loss = loss_fn(vf_t, return_t)
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        return actor_loss.detach().cpu().numpy(), vf_loss.detach().cpu().numpy()


if __name__ == "__main__":
    # create agent
    agent = PGAgent(state_size, action_size, actor_lr, value_function_lr, discount)

    for ep in range(episodes):
        env.reset()
        state = env.state
        state_list, mask_list, action_list, reward_list = [], [], [], []

        # stopping condition for training if agent reaches the amount of reward
        if len(stats_rewards_list) > stats_every and np.mean(stats_rewards_list[-stats_every:], axis=0)[1] > 190:
            print("Stopping at episode {} with average rewards of {} in last {} episodes".
                  format(ep, np.mean(stats_rewards_list[-stats_every:], axis=0)[1], stats_every))
            break

        # train in each episode until episode is done
        while True:
            timesteps += 1
            # env.render()
            mask = 1 - env.fog
            action = agent.select_action(state, mask)  # flatten

            # enter action into the env
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            episode_length += 1
            # store agent's trajectory
            state_list.append(state)
            mask_list.append(mask)
            action_list.append(action)
            reward_list.append(reward * reward_scale)

            if done:
                actor_loss, vf_loss = agent.train(state_list, mask_list, action_list, reward_list)
                stats_rewards_list.append((ep, total_reward, episode_length))
                stats_actor_loss += actor_loss
                stats_vf_loss += vf_loss
                total_reward = 0
                episode_length = 0
                if ep % stats_every == 0:
                    print('Episode: {}'.format(ep),
                          'Timestep: {}'.format(timesteps),
                          'Total reward: {:.1f}'.format(np.mean(stats_rewards_list[-stats_every:], axis=0)[1]),
                          'Episode length: {:.1f}'.format(np.mean(stats_rewards_list[-stats_every:], axis=0)[2]),
                          'Actor Loss: {:.4f}'.format(stats_actor_loss / stats_every),
                          'VF Loss: {:.4f}'.format(stats_vf_loss / stats_every))
                    stats_actor_loss, stats_vf_loss = 0., 0.
                break

            state = next_state

    agent.save_checkpoints(5000)


    games_no = 1000
    state = env.state
    # mask = env.fog
    wins = 0
    i = 0
    step = 0
    first_loss = 0
    while (i < games_no):
        step += 1
        mask = 1 - env.fog
        action = agent.select_action(state, mask)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        print(mask, i, reward)
        state = next_state
        if (done):
            if (step == 1 and reward == -1):
                first_loss += 1
            i += 1
            env.reset()
            state = env.state
            if (reward == 1):
                wins += 1
            step = 0

    ### First_loss is subtracted so that the games with first pick as bomb are subtracted
    print("Model: {}".format("REINFORCE"))
    print("Win Rate: " + str(wins * 100 / (games_no)))
    print("Win Rate excluding First Loss: " + str(wins * 100 / (games_no - first_loss)))



