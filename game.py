import time
import numpy as np

class MineSweeper():
    def __init__(self,width,height,bomb_no):

        ### grid contains the values of entire map :    [0 = nothing,    -1 = bomb,     1,2,3... = hints of bombs nearby]
        ### fog contains the things that are visible to the player/ system :        [   0 = not visible,    1 = visible ]
        ### grid_width, grid_height, bomb_no are self explanatory
        ### bomb_locs contains *flattened* locations of the bombs in the grid

        ### Initialize game parameters
        self.grid_width = width
        self.grid_height = height
        self.bomb_no = bomb_no
        self.box_count = self.grid_width*self.grid_height
        self.uncovered_count = 0
        self.reset()

    def reset(self):

        ### Initialize the grid
        self.grid = np.zeros((self.grid_width,self.grid_height),dtype=np.int)
        self.fog = np.zeros((self.grid_width,self.grid_height),dtype=np.int)
        self.state = np.zeros((self.grid_width,self.grid_height),dtype=np.int)
    
        self.bomb_locs = np.random.randint(self.box_count, size=self.bomb_no)

        ### Plants the bombs and creates the solutions
        self.plant_bombs()
        self.hint_maker()
        self.update_state()

        self.uncovered_count = 0

    def update_state(self):
        self.state = np.multiply(self.grid,self.fog)
        self.state = np.add(self.state,(self.fog-1))

    def print_state(self):
        self.print_matrix(self.state)

    def print_matrix(self,matrix):
        for row in matrix:
            to_print = ""
            for ele in row:
                to_print+=str(ele)+"\t"
            print(to_print)

    def plant_bombs(self):
        reordered_bomb_locs = []
        for bomb_loc in self.bomb_locs:
            row = int(bomb_loc/self.grid_width)
            col = int(bomb_loc%self.grid_width)
            self.grid[row][col] = -1
            reordered_bomb_locs.append((row,col))
        self.bomb_locs = reordered_bomb_locs
    
    def hint_maker(self):
        for r,c in self.bomb_locs:
            for i in range(r-1,r+2):
                for j in range(c-1,c+2):
                    if(i>=0 and j>=0 and i<self.grid_height and j<self.grid_width and self.grid[i][j]!=-1):
                        self.grid[i][j]+=1
    
    def get_state(self):
        return self.state.flatten()

    def unfog_zeros(self,i,j):
        queue = []
        queue.append((i,j))
        while(len(queue)>0):
            i,j = queue.pop()
            for r in range(i-1,i+2):
                for c in range(j-1,j+2):
                    if(r>=0 and r<self.grid_height and c>=0 and c<self.grid_width):
                        if(self.grid[r][c]==0 and self.fog[r][c]==0):
                            queue.append((r,c))
                        self.fog[r][c]=1
                        self.uncovered_count+=1

    def choose(self,i,j):

        ### If selects something already selected/ uncovered
        if(self.fog[i][j]==1):
            return self.state,False,-0.3

        ### If selected tile has a 0 under it
        if(self.grid[i][j]==0):
            self.unfog_zeros(i,j)
            self.update_state()
            if(self.uncovered_count==self.box_count-self.bomb_no):
                return self.state,True,1
            return self.state,False,0.5

        ### If selected tile has something > 0 under it
        elif(self.grid[i][j]>0):
            self.update_state()
            if(self.uncovered_count==self.box_count-self.bomb_no):
                return self.state,True,1
            return self.state,False,0.5

        ### If selected tile has a bomb under it
        else:
            return self.state,True,-1

def speed_test(iterations):
    start = time.perf_counter()
    for i in range(iterations):
        game = MineSweeper(10,10,10,False)
        game.choose(0,0)
    end = time.perf_counter()-start
    return end

# iterations = 2500
# print("Time taken for "+str(iterations)+" steps is "+str(speed_test(iterations)))
