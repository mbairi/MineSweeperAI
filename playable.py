from game import MineSweeper
from renderer import Render
import numpy as np 
import time
import pygame

class Play():
    def __init__(self):
        self.width = 20
        self.height = 20
        self.bombs = 20
        self.env = MineSweeper(self.width,self.height,self.bombs)
        self.renderer = Render(self.env.state)
        self.renderer.state = self.env.state

    def do_step(self,i,j):
        i=int(i/30)
        j=int(j/30)
        next_state,terminal,reward = self.env.choose(i,j)
        self.renderer.state = self.env.state
        self.renderer.draw()
        return next_state,terminal,reward

def main():
    play = Play()
    play.renderer.draw()
    print(play.env.grid)
    while(True):
        events = play.renderer.bugfix()
        for event in events:
            if(event.type==pygame.MOUSEBUTTONDOWN):
                y,x = pygame.mouse.get_pos()
                _,terminal,reward= play.do_step(x,y)
                print(reward)
                print(play.env.uncovered_count)
                if(terminal):
                    if(reward==-1):
                        print("EZ LOSS")
                    else: 
                        print("EZ CLAP")
                    play.env.reset()
                    play.renderer.state=play.env.state
                    play.renderer.draw()
                    print(play.env.grid)
                    
main()