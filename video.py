import cv2
import numpy as np
import pyscreenshot as ImageGrab



class VideoRecorder:

    def __init__(self):
        # Define the codec and create VideoWriter object
        self.fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.out = cv2.VideoWriter('./videos/output.avi', self.fourcc, 40.0, (160, 210))

    def record(self, img):

        #img = ImageGrab.grab()
        img_np = np.array(img)

        self.out.write(img_np)
        cv2.imshow("Screen", img_np)

    def stop(self):
        self.out.release()
        cv2.destroyAllWindows()