import pygame, sys
from pygame.locals import *
from keras.models import load_model
import numpy as np
import cv2

WINSIZEX = 640
WINSIZEY = 480

WHITE = (255, 255, 255)
BLACK = (0,0,0)
RED = (255, 0, 0)

IMAGESAVE = False

PREDICT = True

model = load_model("bestmodel.h5")

LABELS = {
    0: "Zero", 1: "One",
    2: "Two", 3: "Three",
    4: "Four", 5: "Five",
    6: "Six", 7: "Seven",
    8: "Eight", 9: "Nine"
}

# initializing our pygame
pygame.init()

FONT = pygame.font.SysFont(None, 36)  # Initialize the font

DISPLAYSURF = pygame.display.set_mode((WINSIZEX, WINSIZEY))

pygame.display.set_caption("Digit Board")

iswriting = False

num_xcord = []
num_ycord = []
imag_cnt = 0

BOUNDRYINC = 5

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)
            num_xcord.append(xcord)
            num_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            if num_xcord and num_ycord:  # Check if the lists are not empty
                num_xcord = sorted(num_xcord)
                num_ycord = sorted(num_ycord)

                rect_min_x, rect_max_x = max(num_xcord[0] - BOUNDRYINC, 0), min(num_xcord[-1] + BOUNDRYINC, WINSIZEX)
                rect_min_Y, rect_max_Y = max(num_ycord[0] - BOUNDRYINC, 0), min(num_ycord[-1] + BOUNDRYINC, WINSIZEY)

                num_xcord = []
                num_ycord = []
            else:
                # Handle the case when no points are recorded
                rect_min_x, rect_max_x = 0, 0
                rect_min_Y, rect_max_Y = 0, 0

            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_Y:rect_max_Y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite("image.png")
                imag_cnt += 1

            if PREDICT:
                image = cv2.resize(img_arr, (28, 28))
                image = np.pad(image, (10, 10), 'constant', constant_values=0)
                image = cv2.resize(image, (28, 28)) / 255

                label = str(LABELS[np.argmax(model.predict(image.reshape(1, 28, 28, 1)))])

                textSurface = FONT.render(label, True, RED, WHITE)
                textRecObj = textSurface.get_rect()  # Use get_rect() to get the bounding rectangle
                textRecObj.left, textRecObj.bottom = rect_min_x, rect_max_Y

                DISPLAYSURF.blit(textSurface, textRecObj)

    pygame.display.update()  # Update the display after rendering text
