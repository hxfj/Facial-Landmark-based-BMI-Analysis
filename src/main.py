from Facial_landmark import regression_model as RGS
from Facial_landmark import classification_model as ClF

import cv2
import numpy as np

def regression():
    rgs = RGS.create_regressionmodel()
    img_y = rgs.SVR_mod(imgpath='./paper_data/f_001.jpg')
    print(img_y)

def classification():
    clf = ClF.create_classificationmodel()
    img_y = clf.RFC_mod(imgpath='./paper_data/f_001.jpg')
    print(img_y)

if __name__ == '__main__':
    regression()
    classification()