import time

from numba import njit
from numpy import intp
from numpy import multiply, zeros, add, random, count_nonzero


class MineSweeper:
    def __init__(self, width, height, bomb_no, rule='default'):

        # grid contains the values of entire map: [0 = nothing, -1 = bomb, 1,2,3... = hints of bombs nearby]
        # fog contains the things that are visible to the player/system: [0 = not visible, 1 = visible]
        # state is the same as grid but all invisible squares = -1, located bombs = -2
        # grid_width, grid_height, bomb_no are self-explanatory
        # bomb_locs contains *flattened* locations of the bombs in the grid
        # rule can be either 'default' (completely random bombs), 'winxp' (no bomb at the first click) or 'win7' (no
        # bomb at nor next to the first click)
        self.grid_width = width
        self.grid_height = height
        self.bomb_no = bomb_no
        self.box_count = self.grid_width * self.grid_height
        assert self.box_count > self.bomb_no, "Too many bombs."
        self.uncovered_count = 0
        self.rule = rule
        assert self.rule in ['default', 'winxp', 'win7'], "Incorrect rule keyword."
        self.grid = None
        self.fog = None
        self.state = None
        self.bomb_locs = None
        self.bomb_planted = False
        self.reset()

    def reset(self):
        self.grid = zeros((self.grid_width, self.grid_height), dtype=intp)
        self.fog = zeros((self.grid_width, self.grid_height), dtype=intp)
        self.state = zeros((self.grid_width, self.grid_height), dtype=intp)
        # under default rule, bombs are planted during reset
        if self.rule == 'default':
            self.bomb_locs = random.choice(range(self.box_count), self.bomb_no, replace=False)
            self._plant_bombs()
            self._hint_maker()
            self._update_state()
            self.uncovered_count = 0
            self.bomb_planted = True
        # under winxp or win7 rule, bombs are planted at the first choose
        else:
            self.bomb_planted = False

    # Updates the state after choosing decision
    def _update_state(self):
        self.state = multiply(self.grid, self.fog)
        self.state = add(self.state, (self.fog - 1))

    def _flatten2grid(self, loc):
        row = loc // self.grid_width
        col = loc % self.grid_width
        return row, col

    def _grid2flatten(self, row, col):
        return self.grid_width * row + col

    def _find_neighbors(self, row, col):
        neighbors = []
        for i in range(row - 1, row + 2):
            for j in range(col - 1, col + 2):
                if -1 < i < self.grid_height and -1 < j < self.grid_width:
                    neighbors.append((i, j))
        neighbors.remove((row, col))
        return neighbors

    # Used during initialization to make bomb areas -1 on grid
    def _plant_bombs(self):
        reordered_bomb_locs = []
        grid_width = self.grid_width
        for bomb_loc in self.bomb_locs:
            row, col = self._flatten2grid(bomb_loc)
            self.grid[row][col] = -1
            reordered_bomb_locs.append((row, col))
        self.bomb_locs = reordered_bomb_locs

    # Used after planting bombs in initialization phase to make hints 1,2,3... bombs nearby etc
    def _hint_maker(self):
        grid_height = self.grid_height
        grid_width = self.grid_width
        for r, c in self.bomb_locs:
            for i, j in self._find_neighbors(r, c):
                if self.grid[i][j] != -1:
                    self.grid[i][j] += 1

    # Game logic for choosing a point in grid
    def choose(self, i, j):
        # plant bombs after the first choose if rule is not default
        if not self.bomb_planted:
            if self.rule == 'winxp':
                candidates = list(range(self.box_count))
                candidates.remove(self._grid2flatten(i, j))
            else:  # == 'win7'
                candidates = list(range(self.box_count))
                for row, col in self._find_neighbors(i, j):
                    candidates.remove(self._grid2flatten(row, col))

            self.bomb_locs = random.choice(candidates, self.bomb_no, replace=False)
            self._plant_bombs()
            self._hint_maker()
            self._update_state()
            self.uncovered_count = 0
            self.bomb_planted = True

        if self.grid[i][j] == 0:
            unfog_zeros(self.grid, self.fog, i, j)
            self.uncovered_count = count_nonzero(self.fog)
            self._update_state()
            if self.uncovered_count == self.box_count - self.bomb_no:
                return self.state, True, 1
            self.auto_labeling()
            return self.state, False, 0.5

        elif self.grid[i][j] > 0:
            self.fog[i][j] = 1
            self.uncovered_count = count_nonzero(self.fog)
            self._update_state()
            if self.uncovered_count == self.box_count - self.bomb_no:
                return self.state, True, 1
            self.auto_labeling()
            return self.state, False, 0.5

        else:
            return self.state, True, -1
        
    def auto_labeling(self):
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                if self.state[i][j] > 0:
                    neighbors = self._find_neighbors(i, j)
                    invisible_nbr_no = [self.state[ni, nj] < 0 for ni, nj in neighbors].count(True)
                    if self.state[i][j] == invisible_nbr_no:
                        for ni, nj in neighbors:
                            self.state[ni][nj] = -2

# THIS is the slightly more complex logic with Breadth First Search
# Used to unfog the grid if zeros are there in nearby region and if the chosen grid is a zero cell
@njit(fastmath=True)
def unfog_zeros(grid, fog, i, j):
    h, w = grid.shape
    queue = []
    queue.append((i, j))
    while len(queue) > 0:
        i, j = queue.pop()
        for r in range(i - 1, i + 2):
            for c in range(j - 1, j + 2):
                if 0 <= r < h and 0 <= c < w:
                    if grid[r][c] == 0 and fog[r][c] == 0:
                        queue.append((r, c))
                    fog[r][c] = 1


def speed_test(iterations):
    start = time.perf_counter()
    for i in range(iterations):
        game = MineSweeper(10, 10, 10)
        game.choose(0, 0)
    end = time.perf_counter() - start
    return end

# iterations = 2500
# print("Time taken for "+str(iterations)+" steps is "+str(speed_test(iterations)))


# if __name__ == "__main__":
#     env = MineSweeper(9, 9, 10, 'win7')
#     print(env._find_neighbors(0, 0))
