import pygame
import numpy as np
import sys 
import random


class Render():
    def __init__(self,state):

        self.GRAY_SHADES= [
                        (39, 43, 48),
                        (34, 40, 49),
                        (238, 238, 238),
                        ]
        
        self.BLUE = [
                    (0, 172, 181),
                    (0,165,181),
                    (0,160,181),
                    (0,155,181),
                    (0,150,181)
                    ]
        h,w = state.shape
        self.blockSize = 30
        self.WINDOW_HEIGHT = h*self.blockSize
        self.WINDOW_WIDTH = w*self.blockSize
        self.state = state
        self.init()
        

    def init(self):
        pygame.init()
        self.font = pygame.font.SysFont('Courier', 18,bold=True)
        self.SCREEN = pygame.display.set_mode((self.WINDOW_HEIGHT, self.WINDOW_WIDTH))
        self.CLOCK = pygame.time.Clock()
        self.SCREEN.fill(self.GRAY_SHADES[1])
    
    def draw(self):
        self.drawGrid()
        pygame.display.update()
    
    def bugfix(self):
        return pygame.event.get()

    def addText(self,no,x,y,color):
        self.SCREEN.blit(self.font.render(str(no), True, color), (x, y))
        pygame.display.update()


    def drawGrid(self):
        
        j=0
        for column in range(0, self.WINDOW_WIDTH, self.blockSize):
            i=0
            for row in range(0, self.WINDOW_HEIGHT, self.blockSize):
                if(self.state[i][j]==-1):
                    pygame.draw.rect(self.SCREEN, self.GRAY_SHADES[0], [column,row,self.blockSize,self.blockSize])
                if(self.state[i][j]==0):
                    pygame.draw.rect(self.SCREEN, self.GRAY_SHADES[2], [column,row,self.blockSize,self.blockSize])
                elif(self.state[i][j]>0):
                    pygame.draw.rect(self.SCREEN, self.BLUE[0], [column,row,self.blockSize,self.blockSize])
                    self.addText(self.state[i][j],column+10,row+7,self.GRAY_SHADES[2])
                i+=1
            j+=1



