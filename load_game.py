import pygame
import sys
from pygame.locals import *


def run_game():
    """
    Load the drawing rectangle for user to draw digits.
    """

    pygame.init()
    surface = pygame.display.set_mode((280, 280)) # can change, but will be easier to convert to 28x28 for model.
    pygame.display.set_caption("Draw a single digit (0-9)")

    # colors
    white = pygame.Color(255, 255, 255)
    black = pygame.Color(0,0,0)

    pen_size = 20 # size of drawing pen

    surface.fill(white)

    working = True
    while(working):

        # allow loop to finish
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.update()

        left_click  = pygame.mouse.get_pressed()[0]
        right_click = pygame.mouse.get_pressed()[2] # 1 would be middle mouse I guess

        px, py = pygame.mouse.get_pos()

        if left_click == 1:
            for x in range(pen_size):
                for y in range(pen_size):
                    surface.set_at((px + x - 10, py + y - 10), black)

        if right_click == 1:
            surface.fill(white)


        if event.type == pygame.KEYUP:
            print("key has been pressed")


            



if __name__ == '__main__':
    run_game()
