import pygame

from game import MineSweeper
from renderer import Render


class Play:
    def __init__(self):
        self.width = 16
        self.height = 16
        self.bombs = 40
        self.env = MineSweeper(self.width, self.height, self.bombs, rule='win7')
        self.renderer = Render(self.env.state)
        self.renderer.state = self.env.state

    def do_step(self, i, j):
        i = int(i / 30)
        j = int(j / 30)
        next_state, terminal, reward = self.env.choose(i, j, auto_flag=True, auto_play=True)
        self.renderer.state = self.env.state
        self.renderer.draw()
        return next_state, terminal, reward


if __name__ == "__main__":
    play = Play()
    play.renderer.draw()
    print(play.env.grid)
    while True:
        events = play.renderer.bugfix()
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                y, x = pygame.mouse.get_pos()
                _, terminal, reward = play.do_step(x, y)
                # print(reward)
                # print(play.env.uncovered_count)
                if terminal:
                    if reward == -1:
                        print("LOSS")
                    else:
                        print("WIN")
                    play.env.reset()
                    play.renderer.state = play.env.state
                    play.renderer.draw()
                print(play.env.state)
