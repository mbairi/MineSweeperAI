import time
import torch
import numpy as np
import torch.nn as nn
from dqn import DQN,CNNDQN
from game import MineSweeper
from torch.autograd import Variable
from dqn import Buffer
from multiprocessing import Process
from renderer import Render


class Driver():
    def __init__(self,width,height,bomb_no,render_flag):

        self.width = width
        self.height = height
        self.bomb_no = bomb_no
        self.box_count = width*height
        self.env = MineSweeper(self.width,self.height,self.bomb_no)
        self.dqn = DQN(self.box_count,128)
        self.buffer = Buffer(100000)
        self.gamma = 0.99
        self.optimizer = torch.optim.Adam(self.dqn.parameters())
        self.render_flag = render_flag
        self.epsilon_decay = 0.001

        if(render_flag):
            self.Render = Render(self.env.state)

    def get_action(self,state):
        state = state.flatten()
        action = self.dqn.act(state)
        return action

    def do_step(self,action):
        i = int(action/self.width)
        j = action%self.width
        next_state,terminal,reward = self.env.choose(i,j)
        if(self.render_flag):
            self.Render.state = self.env.state
            self.Render.draw()
            self.Render.bugfix()
        return next_state,terminal,reward
    
    def TD_Loss(self):
        state,action,reward,next_state,terminal = self.buffer.sample(4096)

        state      = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)), requires_grad=False)
        action     = Variable(torch.LongTensor(np.float32(action)))
        reward     = Variable(torch.FloatTensor(reward))
        done       = Variable(torch.FloatTensor(terminal))

        q_values      = self.dqn(state)
        next_q_values = self.dqn(next_state)
        
        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        
        loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

        loss_print = loss.item()    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss_print

def main():
    driver = Driver(10,10,10,False)
    state = driver.env.state
    count = 0
    running_reward = 0 
    batch_no = 0
    epochs = 1200
    
    log = open("./Logs/dnn_log.txt",'w')
    
    while(batch_no<epochs):

        action = driver.get_action(state)
        next_state,terminal,reward = driver.do_step(action)
        driver.buffer.push(state.flatten(),action,reward,next_state.flatten(),terminal)
        state = next_state
        count+=1
        running_reward+=reward

        if(terminal):
            driver.env.reset()
            state = driver.env.state

        if(count==4096):
            driver.TD_Loss()
            batch_no+=1
            print("Batch: "+str(batch_no)+"\tAvg Reward: "+str(running_reward/4096)+"\tEpsilon: "+str(driver.dqn.epsilon))
            log.write(str(running_reward/4096)+"\n")
            driver.dqn.epsilon = max(0.01,driver.dqn.epsilon-driver.epsilon_decay)
            running_reward=0
            count=0
    
    torch.save(driver.dqn.state_dict(),"./pre-trained/dqn_dnn.pth")

main()