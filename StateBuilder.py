from scipy.misc import imresize
import numpy as np

class StateBuilder:

    def __init__(self):
        lkj = 1

    def preprocess(self, observation):

        observation_gray_cropped = np.zeros((158, 144))

        observation_gray = observation * [0.299, 0.587, 0.114]  # gray-scale

        notNullRGBIndexArray = observation_gray.nonzero()
        length = notNullRGBIndexArray[0].__len__()

        for i in range(0, length, 3):  # get every 4th index, because they're represent a new pixel
            x = notNullRGBIndexArray[1][i]
            y = notNullRGBIndexArray[0][i]

            if (31 < y < 190 and 7 < x < 152):  # cropping
                observation_gray_cropped[(y - 32)][(x - 8)] = observation_gray[y][x][0] + observation_gray[y][x][1] + \
                                                              observation_gray[y][x][2]

        numpy_matrix = imresize(observation_gray_cropped, (84, 84))  # down-scale

        return numpy_matrix