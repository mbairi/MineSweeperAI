import time
import torch
import numpy as np
import sys
sys.path.insert(1,"./Models")
import torch.nn as nn
from ddqn import DDQN
from game import MineSweeper
from torch.autograd import Variable
from ddqn import Buffer
from multiprocessing import Process
from renderer import Render
from numpy import float32
from torch import FloatTensor,LongTensor

class Driver():
    def __init__(self,width,height,bomb_no,render_flag):

        self.width = width
        self.height = height
        self.bomb_no = bomb_no
        self.box_count = width*height
        self.env = MineSweeper(self.width,self.height,self.bomb_no)
        self.current_model = DDQN(self.box_count,self.box_count)
        self.target_model = DDQN(self.box_count,self.box_count)
        self.optimizer = torch.optim.Adam(self.current_model.parameters(),lr=0.001,weight_decay=1e-5)
        self.target_model.load_state_dict(self.current_model.state_dict())
        self.buffer = Buffer(50000)
        self.gamma = 0
        self.render_flag = render_flag
        self.epsilon_decay = 0.0004
        self.epsilon_min = 0.00001

        if(render_flag):
            self.Render = Render(self.env.state)

    def get_action(self,state,mask):
        state = state.flatten()
        mask = mask.flatten()
        action = self.current_model.act(state,mask)
        return action

    def do_step(self,action):
        i = int(action/self.width)
        j = action%self.width
        next_state,terminal,reward = self.env.choose(i,j)
        next_fog = 1-self.env.fog
        return next_state,terminal,reward,next_fog
    
    def TD_Loss(self):
        state,action,mask,reward,next_state,next_mask,terminal = self.buffer.sample(2048)
        state      = Variable(FloatTensor(float32(state)))
        next_state = Variable(FloatTensor(float32(next_state)), requires_grad=False)
        action     = Variable(LongTensor(float32(action)))
        mask      = Variable(FloatTensor(float32(mask)))
        next_mask      = Variable(FloatTensor(float32(next_mask)))
        reward     = Variable(FloatTensor(reward))
        done       = Variable(FloatTensor(terminal))
        q_values      = self.current_model(state,mask)
        next_q_values = self.target_model(next_state,next_mask)
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

    driver = Driver(6,6,6,False)
    state = driver.env.state
    count = 0
    running_reward = 0 
    batch_no = 0
    epochs = 20000
    wins=0
    losses =0 
    total=0
    log = open("./Logs/ddqn_log.txt",'w')

    while(batch_no<epochs):

        mask = 1- driver.env.fog
        action = driver.get_action(state,mask)
        next_state,terminal,reward,_ = driver.do_step(action)
        driver.buffer.push(state.flatten(),action,mask.flatten(),reward,next_state.flatten(),(1-driver.env.fog).flatten(),terminal)
        
        state = next_state
        count+=1
        running_reward+=reward

        if(terminal):
            if(reward==1):
                wins+=1
            driver.env.reset()
            state = driver.env.state
            mask = driver.env.fog
            total+=1

        if(count==2048):
            loss = driver.TD_Loss()
            batch_no+=1
            avg_reward = running_reward/2048
            wins = wins*100/total
            res = [
                    str(batch_no),
                    "\tAvg Reward: ",
                    str(avg_reward),
                    "\t Loss: ",
                    str(loss),
                    "\t Wins: ", 
                    str(wins),
                    "\t Epsilon: ",
                    str(driver.current_model.epsilon)
            ]
            log_line = " ".join(res)
            print(log_line)
            log.write(log_line)
            driver.current_model.epsilon = max(driver.epsilon_min,driver.current_model.epsilon-driver.epsilon_decay)
            running_reward=0
            count=0
            wins=0
            total=0
            if(batch_no%1000==0):
                path = "./pre-trained/ddqn_dnn"+str(batch_no)+".pth"
                torch.save({
                    'epoch': batch_no,
                    'current_state_dict': driver.current_model.state_dict(),
                    'target_state_dict' : driver.target_model.state_dict(),
                    'optimizer_state_dict': driver.optimizer.state_dict(),
                }, path)

main()