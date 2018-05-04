import cv2
import numpy as np
import datetime


class VideoRecorder:

    def __init__(self, config):
        # Define the codec and create VideoWriter object
        self.fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.out = cv2.VideoWriter('./videos/'+config + '_Day_' + datetime.date.today().strftime("%j") + '.avi', self.fourcc, 30.0, (160, 210))

    def record(self, img):
        img_np = np.array(img)
        self.out.write(img_np)

    def stop(self):
        self.out.release()