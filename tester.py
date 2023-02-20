import time

import torch
import numpy as np
import sys
sys.path.insert(1,"./Models")
from ddqn import  DDQN
from dqn import DQN
from ppo import PPO
from AC0 import AC0
import copy
from scipy.special import comb
 
from renderer import Render
from game import MineSweeper
from renderer import Render

model_list={"DDQN":DDQN, "DQN":DQN, "PPO":PPO, "AC0":AC0}



### Preferably don't mess with the parameters for now.
### Class takes in only one parameter as initialization, render true or false
class Tester():
    def __init__(self,render_flag,model_type):       
        self.model_type=model_type
        self.model = model_list[self.model_type](36,36,cuda=False)
        # self.model = DQN(36,36)
        self.render_flag = render_flag
        self.width = 6
        self.height = 6
        self.env = MineSweeper(self.width, self.height, 6, rule='win7')
        if (self.render_flag):
            self.renderer = Render(self.env.state)
        self.load_models(20000)

    def grid2flatten(self, row, col):
        return self.width * row + col
    
    def get_action(self,state):
        state = state.flatten()
        mask = (1 - self.env.fog).flatten()
        action = self.model.act(state, mask)
        return action

    def get_definite_action(self,state):

        Decider=DefiniteDecision(self.env,state)
        state_now, possible_mines=Decider.decision()
        if possible_mines==None or len(possible_mines)==0:
            return False, -1

        
        new_miner_loc=possible_mines[0][0]==2
        while new_miner_loc:
            state_now[possible_mines[0][1]][possible_mines[0][2]]=-2
            Decider=DefiniteDecision(self.env,state_now)
            state_now, possible_mines=Decider.decision()
            if possible_mines==None or len(possible_mines)==0:
                break
            new_miner_loc=possible_mines[0][0]==2

        
        flag, coords=Decider.make_decision(possible_mines)

        if flag:
            return True, self.grid2flatten(coords[0],coords[1])
        
        return False, -1

        
        
    def load_models(self,number):
        path = "pre-trained/"+self.model_type.lower()+"_dnn"+str(number)+".pth"
        # path = "pre-trained/dqn_dnn"+str(number)+".pth"
        dict = torch.load(path)
        self.model.load_state(dict)
        self.model.epsilon = 0

    def do_step(self, action):
        i = int(action / self.width)
        j = action % self.width

        if (self.render_flag):
            self.renderer.state = self.env.state
            self.renderer.draw()
            self.renderer.bugfix()
        next_state, terminal, reward = self.env.choose(i, j)
        return next_state, terminal, reward


class DefiniteDecision():
    def __init__(self, env, state):
       
        # self.info=state
        self.height = env.grid_width
        self.width = env.grid_height
        self.mineNumber = env.bomb_no
        # self.doFirstFlip = len(env.bomb_locs)

        self.mineField = copy.deepcopy(state)
        self.state = "start"
        self.unknown = 1-env.fog # [1 = not visible, 0 = visible]
        self.restMine = env.bomb_no

        self.ratio = 1.5
        self.cut=10

    def translate(self,state):
        pass


    def decision(self):
        np.random.seed(int(time.time()))
        '''
         '' = -1 grids undiscovered
         '*' = -2 grids must have bombs
         '?' = -3 grids might have bombs
        '''
        
  
        self.possibility = np.zeros(shape=(self.height, self.width))

        # if not self.doFirstFlip and self.state == "start":
        #     self.state = self.infoState
        #     return [[1, np.random.randint(low=0, high=self.height), np.random.randint(low=0, high=self.width)]]
        # self.state = self.infoState

        if not self.restMine:
            allRestPosition = []
            for i in range(self.height):
                for j in range(self.width):
                    if self.mineField[i][j] == -1:
                        allRestPosition.append([1, i, j])
            return self.mineField, allRestPosition

        restAction = []

        

        for i in range(self.height):
            for j in range(self.width):
                if 1 <= self.mineField[i][j] <= 9:
                    closeBlocks = self.getAllCloseBlock(i, j)
                    contentBlocks = [self.mineField[i][j] for i, j in closeBlocks]
                    if -1 in contentBlocks:
                        if contentBlocks.count(-2) == int(self.mineField[i][j]):
                            restAction.append([3, i, j])
                            for i, j in closeBlocks:
                                if self.mineField[i][j] == -1:
                                    restAction.append([1, i, j])
        if restAction:
            
            return self.mineField, restAction


        for i in range(self.height):
            for j in range(self.width):
                if 1 <= self.mineField[i][j] <= 9:
                    availablePosition, lack = self.checkNumber(i, j)
                    if lack:
                        if len(availablePosition) == lack:
                            allDecision = []
                            for l in range(lack):
                                if self.mineField[availablePosition[l][0]][availablePosition[l][1]] == -1:
                                    allDecision.append([2, availablePosition[l][0], availablePosition[l][1]])
                                    self.restMine -= 1
                            for l in range(lack):
                                cb=self.getAllCloseBlock(availablePosition[l][0], availablePosition[l][1])
                                for b in cb:
                                    if 1 <= self.mineField[b[0]][b[1]] <= 9:
                                        allDecision.append([3, b[0], b[1]])

                            return self.mineField, allDecision

        return self.mineField, []
        # return self.globalDecision()

    def make_decision(self, actions):
        mine_temp=copy.deepcopy(self.mineField)
        if len(actions)==0:
            return False,[-1,-1]
        
        actions=sorted(actions, key=lambda item:item[0])

        for a in actions:
            if a[0]==1:
                return True, [a[1],a[2]]
            elif a[0]==2:
                mine_temp[a[1],a[2]]=-2
            elif a[0]==3:
                i=a[1]
                j=a[2]

                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0 or not 0 <= i + di < self.height or not 0 <= j + dj < self.width:
                            continue
                        closeBlocks = self.getAllCloseBlock(i, j)
                        contentBlocks = [self.mineField[i][j] for i, j in closeBlocks]
                        if -1 in contentBlocks:
                            if contentBlocks.count(-2) == int(self.mineField[i][j]):
                                remaining=[[a,b] for a,b in closeBlocks if self.mineField[a,b]==-1]
                                if len(remaining):
                                    return True, [remaining[0][0],remaining[0][1]]

                        
        return False, (-1,-1)


    def checkNumber(self, i, j):
        lack = int(self.mineField[i][j])
        availablePosition = []
        for b in self.getAllCloseBlock(i, j):
            if self.mineField[b[0]][b[1]] == -1:
                availablePosition.append(b)
            elif self.mineField[b[0]][b[1]] == -2:
                lack -= 1
        return availablePosition, lack

    def getAllCloseBlock(self, i, j):
        blockList = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if not di and not dj:
                    continue
                if i + di < 0 or i + di >= self.height:
                    continue
                if j + dj < 0 or j + dj >= self.width:
                    continue
                blockList.append([i + di, j + dj])
        return blockList



### Tests winrate in "games_no" games
def win_tester(games_no,model_type,use_definite=True):
    tester = Tester(False,model_type)
    state = tester.env.state
    mask = tester.env.fog
    wins = 0
    i = 0
    step = 0
    first_loss = 0
    start=True
    method="AI"
    while (i < games_no):
        step += 1
       
        if use_definite:
            if start:
                action = tester.get_action(state) 
                method="AI"
                start=False
            else:
                s1=copy.deepcopy(state)
                has_definite_action, act = tester.get_definite_action(s1)
                if has_definite_action:
                    action=act
                    method="definite"
                else:
                    action = tester.get_action(state) 
                    method="AI"

        
        else:
            action = tester.get_action(state) 
            method="AI"

        next_state, terminal, reward = tester.do_step(action)        
        state = next_state

        if (terminal):
            
            if (step == 1 and reward == -1):
                first_loss += 1
            i += 1
            tester.env.reset()
            state = tester.env.state
            if (reward == 1):
                wins += 1

            # if reward==-1:
            #     print(method)
            step = 0

        # break
        

    ### First_loss is subtracted so that the games with first pick as bomb are subtracted
    print("Model: {}".format(model_type))
    print("Win Rate: "+str(wins*100/(games_no)))
    print("Win Rate excluding First Loss: "+str(wins*100/(games_no-first_loss)))


def slow_tester(model_type):    
    tester = Tester(True,model_type)
    state = tester.env.state
    count = 0
    start = time.perf_counter()
    step = 0
    first_loss = 0

    while (True):
        count += 1
        step += 1
        action = tester.get_action(state)
        next_state, terminal, reward = tester.do_step(action)
        state = next_state
        print(reward)
        time.sleep(0.5)

        if (terminal):
            if (reward == 1):
                print("WIN")
            else:

                print("LOSS")
            tester.env.reset()
            step = 0
            state = tester.env.state
            


def main():
    model_type="DDQN"
    win_tester(1000,model_type,use_definite=False)

    win_tester(1000,model_type,use_definite=True)

    # slow_tester(model_type)
main()
