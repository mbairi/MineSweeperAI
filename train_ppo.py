import time
import torch
import numpy as np
import sys
sys.path.insert(1,"./Models")
import torch.nn as nn
from ppo import Actor, Critic, Buffer
from game import MineSweeper
from renderer import Render
from numpy import float32
from torch.autograd import Variable
from multiprocessing import Process
from torch import FloatTensor,LongTensor

'''
GAME PARAMS:
width   = width of board
height  = height of board
bomb_no = bombs on map
env     = minesweeper environment created from "game.py" with params

AI PARAMS:
optimizer:
    lr          = learning rate at 0.002, weight decay 1e-5
    scheduler   = reduces learning rate to 0.95 of itsemf every 2000 steps
buffer  = stores the State, Action, Reward, Next State, Terminal?, and Masks for each state
gamma   = weightage of reward to future actions
epsilon = the randomness of the DDQN agent
    this is not decayed by linear or exponential methods, 
    RBED is used ( Reward Based Epsilon Decay )
        if reward_threshold is exceeded, epsilon becomes 0.9x of itself
        and next the reward_threshold is increased by reward_step
batch_size = set to 2048 decisions before each update

Model Details:
I have made a pretty small model so that it executes fast and I can reiterate my parameters manually faster
DDQN with epsilon starting at 1 and reduces based on RBED
Has feature extractor layer
Has 2 heads to the model
    advantage and value
    combination of these 2 will give the q value of the state
IMPORTANT : I HAVE ADDED ACTION MASKING, WHICH IMPROVES PERFORMANCE


main() function PARAMS:
save_every : Saves the model every x steps
update_targ_every : Updates the target model to current model every x steps
    (Note I have to try interpolated tau style instead of hard copy)
epochs: self explanatory
logs: Win Rate, Reward, Loss and Epsilon are written to this file and can be visualized using ./Logs/plotter.py
'''

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
device="cpu"
print(device)

class Driver():

    def __init__(self,width,height,bomb_no,render_flag):

        self.width = width
        self.height = height
        self.bomb_no = bomb_no
        self.box_count = width*height
        self.env = MineSweeper(self.width,self.height,self.bomb_no)
        self.actor = Actor(self.box_count,self.box_count).to(device)
        self.critic = Critic(self.box_count).to(device)
        # self.target_model.eval()
        # self.optimizer_actor = torch.optim.Adam(self.actor.parameters(),lr=0.003,weight_decay=1e-5)
        # self.optimizer_critic = torch.optim.Adam(self.critic.parameters(),lr=0.003,weight_decay=1e-5)

        self.optimizer = torch.optim.Adam([
                        {'params': self.actor.parameters(), 'lr': 0.003, 'weight_decay': 1e-5},
                        {'params': self.critic.parameters(), 'lr': 0.003, 'weight_decay': 1e-5}
                    ])
        self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=2000,gamma=0.95)
        # self.scheduler_actor = torch.optim.lr_scheduler.StepLR(self.optimizer_actor,step_size=2000,gamma=0.95)
        # self.scheduler_critic = torch.optim.lr_scheduler.StepLR(self.optimizer_critic,step_size=2000,gamma=0.95)
        # self.target_model.load_state_dict(self.current_model.state_dict())
        self.buffer = Buffer(10000)
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.render_flag = render_flag
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.90
        self.reward_threshold = 0.12
        self.reward_step = 0.01
        self.batch_size = 1024
        self.tau = 5e-5
        self.policy_clip=0.2
        self.mse=nn.MSELoss()
        self.log = open("./Logs/ppo_log.txt",'w')

        if(self.render_flag):
            self.Render = Render(self.env.state)

    
    def load_models(self,number):
        path = "./pre-trained/ppo_dnn"+str(number)+".pth"
        weights = torch.load(path)
        self.actor.load_state_dict(weights['actor_state_dict'])
        self.critic.load_state_dict(weights['critic_state_dict'])
        self.optimizer.load_state_dict(weights['optimizer_state_dict'])      
        self.actor.epsilon = weights['epsilon_actor']
        self.critic.epsilon = weights['epsilon_critic']


    ### Get an action from the DDQN model by supplying it State and Mask
    def get_action(self,state,mask):
        state = state.flatten()
        mask = mask.flatten()

        dist = self.actor(state,mask)
        value = self.critic(state)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()
        
        return action, probs, value

    ### Does the action and returns Next State, If terminal, Reward, Next Mask
    def do_step(self,action):
        i = int(action/self.width)
        j = action%self.width
        if(self.render_flag):
            self.Render.state = self.env.state
            self.Render.draw()
            self.Render.bugfix()
        next_state,terminal,reward = self.env.choose(i,j)
        next_fog = 1-self.env.fog
        return next_state,terminal,reward,next_fog
    
    ### Reward Based Epsilon Decay 
    def epsilon_update(self,avg_reward):
        if(avg_reward>self.reward_threshold):
            self.actor.epsilon = max(self.epsilon_min,self.actor.epsilon*self.epsilon_decay)
            self.critic.epsilon = max(self.epsilon_min,self.actor.epsilon*self.epsilon_decay)
            self.reward_threshold+= self.reward_step
    
    def TD_Loss(self):
        ### Samples batch from buffer memory
        state,next_state,prob,val,action,mask,reward,terminal = self.buffer.sample(self.batch_size)

        ### Converts the variabls to tensors for processing by DDQN
        state      = Variable(FloatTensor(float32(state))).to(device)
        next_state      = Variable(FloatTensor(float32(next_state))).to(device)
        prob      =  FloatTensor(prob).to(device)
        val       =  FloatTensor(val).to(device)
        mask      = Variable(FloatTensor(float32(mask))).to(device)        
        action     = LongTensor(float32(action)).to(device)        
        reward     = FloatTensor(reward).to(device)
        done       = FloatTensor(terminal).to(device)

        reward = (reward - reward.mean())/(reward.std() + 1e-10)
        with torch.no_grad():
            target_v = reward + self.gamma * self.critic(next_state)

        advantage=np.zeros(len(reward),dtype=np.float32)
        for t in range(len(reward)-1):
            discount=1
            a_t=0
            for k in range(t, len(reward)-1):
                a_t+=discount*(reward[k].item()+self.gamma*val[k+1].item()*(1-int(done[k].item()))-val[k].item())
                discount*=self.gamma*self.gae_lambda
            advantage[t]=a_t
        advantage=torch.tensor(advantage).to(device)

        # advantage = (target_v - self.critic(state)).detach()

        #SGD
        dist=self.actor(state,mask)
        critic_value=self.critic(state)
        critic_value=torch.squeeze(critic_value)
        new_probs=dist.log_prob(action)
        prob_ratio = torch.exp(new_probs - prob)

        weighted_probs=advantage*prob_ratio
        weighted_clipped_probs=torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage
        
        actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
        returns = advantage + val
        critic_loss = (returns-critic_value)**2
        critic_loss = critic_loss.mean()

        total_loss=actor_loss+0.5*critic_loss

        loss_print = total_loss.item()    

        # Propagates the Loss
        self.optimizer.zero_grad()
        total_loss.backward()

        self.optimizer.step()
        self.scheduler.step()

        # for target_param, local_param in zip(self.actor.parameters(), self.critic.parameters()):
        #     target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
        return loss_print

    def save_checkpoints(self,batch_no):
        path = "./pre-trained/ppo_dnn"+str(batch_no)+".pth"
        torch.save({
            'epoch': batch_no,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict' : self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon_actor':self.actor.epsilon,
            'epsilon_critic':self.critic.epsilon
        }, path)

    def save_logs(self,batch_no,avg_reward,loss,wins):
        res = [
                    str(batch_no),
                    "\tAvg Reward: ",
                    str(avg_reward),
                    "\t Loss: ",
                    str(loss),
                    "\t Wins: ", 
                    str(wins),
                    "\t Epsilon_actor: ",
                    str(self.actor.epsilon),
                    "\t Epsilon_critic: ",
                    str(self.critic.epsilon)
        ]
        log_line = " ".join(res)
        print(log_line)
        self.log.write(log_line+"\n")
        self.log.flush()


def main():

    driver = Driver(6,6,6,False)
    state = driver.env.state
    epochs = 20000*4
    save_every = 1000*4
    count = 0
    running_reward = 0 
    batch_no = 0
    wins=0
    total=0
    
    while(batch_no<epochs):
        
        # simple state action reward loop and writes the actions to buffer
        mask = 1- driver.env.fog
        action, probs, value = driver.get_action(state,mask)
        next_state,terminal,reward,_ = driver.do_step(action)

        driver.buffer.push(state.flatten(),next_state.flatten(),probs, value, action,mask.flatten(),reward,terminal)
        state = next_state
        count+=1
        running_reward+=reward

        # Used for calculating winrate for each batch
        if(terminal):
            if(reward==1):
                wins+=1
            driver.env.reset()
            state = driver.env.state
            mask = driver.env.fog
            total+=1

        if(count==driver.batch_size):
            # Computes the Loss
            driver.actor.train()
            driver.critic.train()
            loss = driver.TD_Loss()
            driver.actor.eval()
            driver.critic.eval()

            # Calculates metrics
            batch_no+=1
            avg_reward = running_reward/driver.batch_size
            wins = wins*100/total
            driver.save_logs(batch_no,avg_reward,loss,wins)

            # Updates epsilon based on reward
            driver.epsilon_update(avg_reward)
            
            # Resets metrics for next batch calculation
            running_reward=0
            count=0
            wins=0
            total=0

            # Saves the model details to "./pre-trained" if 1000 batches have been processed
            if(batch_no%save_every==0):
                driver.save_checkpoints(batch_no)

main()