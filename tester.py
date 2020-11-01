import torch
import numpy as np
from dqn import DQN,CNNDQN
from renderer import Render
from game import MineSweeper

class Tester():
    def __init__(self,model):
        if(model=="dnn"):
            self.test_model = DQN(100,128)
        elif(model=="cnn"):
            self.test_model = CNNDQN(10)
        
        self.model_type = model
        self.env = MineSweeper(10,10,10)
        self.renderer = Render(self.env.state)

    def get_action(self,state):
        if(self.model_type=="dnn"):
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
    

def main():
    


        
