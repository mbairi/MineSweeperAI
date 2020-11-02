import torch
import numpy as np
from dqn import DQN,CNNDQN
from renderer import Render
from game import MineSweeper
import time

class Tester():
    def __init__(self,model):
        if(model=="dnn"):
            self.dqn = DQN(100,256)
        elif(model=="cnn"):
            self.dqn = CNNDQN(10)

        self.render_flag = True
        self.model_type = model
        self.width = 10
        self.height = 10
        self.env = MineSweeper(self.width,self.height,10)
        if(self.render_flag):
            self.renderer = Render(self.env.state)
        

    def get_action(self,state):
        if(self.model_type=="dnn"):
            state = state.flatten()
        action = self.dqn.act(state)
        return action
    
    def do_step(self,action):
        i = int(action/self.width)
        j = action%self.width
        if(self.render_flag):
            self.renderer.state = self.env.state
            self.renderer.draw()
            self.renderer.bugfix()
        next_state,terminal,reward = self.env.choose(i,j)
        return next_state,terminal,reward
    

def win_tester():
    tester = Tester("dnn")
    state = tester.env.state
    games = 1000
    wins =0
        
    for i in range(games):
        action = tester.get_action(state)
        next_state,terminal,reward = tester.do_step(action)
        state = next_state
        
        if(terminal):
            tester.env.reset()
            state = tester.env.state
            if(reward==1):
                wins+=1
    
    print(wins/games)


def slow_tester():
    tester = Tester("dnn")
    state = tester.env.state
    count = 0
    start = time.perf_counter()
    while(True):
        count+=1
        action = tester.get_action(state)
        next_state,terminal,reward = tester.do_step(action)
        state = next_state
        print(reward)

        if(reward!=-0.3):
            time.sleep(1)


        if(terminal):
            tester.env.reset()
            state = tester.env.state
            break
        
        

def main():
    #win_tester()
    slow_tester()

main()
        
