from Facial_landmark import regression_model as RGS
from Facial_landmark import classification_model as ClF

import cv2
import numpy as np

def regression():# SVR for example
    rgs = RGS.create_regressionmodel()
    img_y = rgs.SVR_mod(imgpath='./src/example/f_001.jpg')
    print('BMI value',img_y)

def classification():# Random Forests for example
    clf = ClF.create_classificationmodel()
    img_y = clf.RFC_mod(imgpath='./src/example/f_001.jpg')
    print('BMI type',img_y)

def read_SVR_modle():# Example of saved model
    rgs = RGS.create_regressionmodel()
    img_y = rgs.SVR_mod(imgpath='./src/example/f_001.jpg',modpath='./src/data/SVR_lm_BMI.pkl')
    print('BMI value',img_y)

if __name__ == '__main__':
    regression()
    classification()
    read_SVR_modle()