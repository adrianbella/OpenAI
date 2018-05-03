from scipy.misc import imresize
import numpy as np
import cv2


class StateBuilder:

    def __init__(self):
        pass

    @staticmethod
    def pre_process_cv2(observation):

        ly, lx, dim = observation.shape

        observation_cropped = observation[ly // 6: - ly // 14, lx // 20: - lx // 20]  # cropping the game area
        observation_gray_cropped = cv2.cvtColor(observation_cropped, cv2.COLOR_RGB2GRAY)  # gray-scale
        return cv2.resize(observation_gray_cropped, (84, 84))  # reshape
