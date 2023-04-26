import numpy as np
import os.path as path
from Facial_landmark import read_data as RD
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier;
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def create_classificationmodel():
    return Classificationmodel()


class Classificationmodel:
    def __init__(self) -> None:
        self.data_x = None
        self.data_y = None


    def get_data(self):
        """
        @brief: Using VIP dataset as an example
        @return:
            self.data_x: Landmarks(feature).
            self.data_y: BMI type(label)
        """
        rd = RD.create_readdata()
        two_up =  path.abspath(path.join(__file__ ,"../../../paper_data"))
        featpath = (two_up+ "/"+'VIP_G_lm.xlsx' )
        bmipath = (two_up + "/" +'VIP_G_BMI.xlsx')
        self.data_x = rd.read_landmark(lm_path=featpath)
        self.data_y = rd.read_bmi(bmi_path=bmipath,type = 'type')
        return self.data_x,self.data_y

    def RFC_mod(self,imgpath):
        """
        @brief: Use SVR to get BMI value
        @param: 
            imgpath: The path of the image that needs to be measured
        @return:
            img_y: Predicted value of BMI
        """
        rd = RD.create_readdata()
        img_x = rd.get_image_x(imgpath)
        if len(img_x) != 0:
            x,y = self.get_data()
            X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size = 0.1)
            rfc = RandomForestClassifier(max_depth=5)
            rfc.fit(X_train,Y_train)
            img_y = rfc.predict(img_x)
            return img_y
        else:
            return 0

    def GNB_mod(self,imgpath):
        rd = RD.create_readdata()
        img_x = rd.get_image_x(imgpath)
        if len(img_x) != 0:
            x,y = self.get_data()
            X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size = 0.1)
            gnb = GaussianNB()
            gnb.fit(X_train, Y_train)
            img_y = gnb.predict(img_x)
            return img_y
        else:
            return 0

    def SVC_mod(self,imgpath):
        rd = RD.create_readdata()
        img_x = rd.get_image_x(imgpath)
        if len(img_x) != 0:
            x,y = self.get_data()
            X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size = 0.1)
            svc = SVC(C=1.0, kernel='linear', degree=3, gamma=1, coef0=0.0, shrinking=True, probability=False,
                      tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None) 
            svc.fit(X_train, Y_train)
            img_y = svc.predict(img_x)
            return img_y
        else:
            return 0
        
    def MLPC_mod(self,imgpath):
        rd = RD.create_readdata()
        img_x = rd.get_image_x(imgpath)
        if len(img_x) != 0:
            x,y = self.get_data()
            X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size = 0.1)
            mlp = MLPClassifier(hidden_layer_sizes=(100,50,30), activation='relu',solver='adam',
                                alpha=0.01,max_iter=500)
            
            mlp.fit(X_train,Y_train)
            img_y = mlp.predict(img_x)
            return img_y
        else:
            return 0

