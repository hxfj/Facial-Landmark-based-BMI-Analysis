from Facial_landmark import regression_model as RGS
import cv2
import numpy as np

rgs = RGS.create_regressionmodel()
img_y = rgs.SVR_mod(imgpath='./paper_data/f_001.jpg')
print(img_y)