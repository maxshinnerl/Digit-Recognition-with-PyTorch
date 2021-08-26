import pygame
import sys
from pygame.locals import *
import torch
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


def run_game():
    """
    Load the drawing rectangle for user to draw digits.
    """

    model = load_model()

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

        # functionality is janky, using a bool to ensure only one call to model on (any) keypress
        if event.type == pygame.KEYDOWN:
            down = True

        if event.type == pygame.KEYUP:
            if down is True:
                # call model here -- get_single_prediction(model, image)
                # TODO: just have to get the surface as a 28x28 array --> len 784 list of pixels.
                # can either just convert to an array if that works, or just save and re-load as an image.
                # latter is probably easier but would probably be slow.  There is functionality for the former but it seems weird.
                # should try doing this in notebook to do like imshow stuff on the surface
                down = False


            # else do nothing


class FeedForward2Layer(nn.Module):
    def __init__(self, input_size=784, hidden_size=64, output_size=10):
        super(FeedForward2Layer, self).__init__()
        
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.i2h(x)
        x = self.relu(x)
        x = self.h2o(x)
        
        return x


def load_model():
    """
    Load saved model from the training notebook
    """
    model = FeedForward2Layer()
    model.load_state_dict(torch.load('feed_forward_model.pth'))

    return model


def get_single_prediction(model, image):
    """
    Note that image is expected in the list format to cohere with the mnist load functions
    i.e. 1d list of length 28*28 = 784 instead of a 28 by 28 array
    """
    model.eval()
    yhat = model(image)
    return(torch.argmax(yhat).item())


if __name__ == '__main__':
    run_game()
