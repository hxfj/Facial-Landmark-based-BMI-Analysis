from Facial_landmark import facial_landmark as FL
import pandas as pd
import numpy as np
import math

def create_readdata():
    return Readdata()

class Readdata:
    def __init__(self) -> None:
        self.temp = []

    def read_temp_landmark(self,t_path = 'org'):
        """
        @brief: get template landmark for pretreatment.
        @param:
            t_path: The path of the template landmarks(xlsx file).
        @return:
            BMI_feat_ref: Preprocessed landmark of template face.
        """
        if t_path == 'org':
            f_lm_ref=pd.read_excel(r'src/data/template_face.xlsx')# The template faces selected in our paper
        else:
            f_lm_ref=pd.read_excel(t_path)# The template faces selected in our paper

        f_lm_ref_id=f_lm_ref['subject ID']#point index
        f_lm_x_ref = f_lm_ref['X']
        f_lm_y_ref = f_lm_ref['Y']
        f_lm_z_ref = f_lm_ref['Z']
        f_lm_x_ref_mean = f_lm_x_ref.mean()
        f_lm_y_ref_mean = f_lm_y_ref.mean()
        f_lm_z_ref_mean = f_lm_z_ref.mean()
        BMI_feat_ref = []
        feat_point = []

        for i in range(f_lm_ref_id.shape[0]):
            if f_lm_ref_id.iloc[i]!=478:
                f_lm_x_point = f_lm_x_ref[i]-f_lm_x_ref_mean
                f_lm_y_point = f_lm_y_ref[i]-f_lm_y_ref_mean
                f_lm_z_point = f_lm_z_ref[i]-f_lm_z_ref_mean
                if f_lm_ref_id.iloc[i] == 5:
                    f_lm_point_5 = (f_lm_x_point,f_lm_y_point,f_lm_z_point)
                if f_lm_ref_id.iloc[i] == 10:
                    f_lm_point_10 = (f_lm_x_point,f_lm_y_point,f_lm_z_point)
                
                pt_pos = [f_lm_x_point,f_lm_y_point,f_lm_z_point]
                feat_point.append(pt_pos)
            else : #Landmarks for each face
                f_lm_x_point = f_lm_x_ref[i]-f_lm_x_ref_mean
                f_lm_y_point = f_lm_y_ref[i]-f_lm_y_ref_mean
                f_lm_z_point = f_lm_z_ref[i]-f_lm_z_ref_mean
                pt_pos = [f_lm_x_point,f_lm_y_point,f_lm_z_point]
                feat_point.append(pt_pos)#
                nose_dist = math.sqrt(math.pow((f_lm_point_5[0] - f_lm_point_10[0]), 2) 
                                    + math.pow((f_lm_point_5[1] - f_lm_point_10[1]), 2)
                                    + math.pow((f_lm_point_5[2] - f_lm_point_10[2]), 2))#Nose length
                if np.array(BMI_feat_ref).size == 0:
                    BMI_feat_ref.append(feat_point)
                    BMI_feat_ref = np.array(BMI_feat_ref)
                    BMI_feat_ref = BMI_feat_ref/nose_dist
                    BMI_feat_ref = BMI_feat_ref.reshape(478,-1)
                    feat_point = []
            self.temp = BMI_feat_ref
        return BMI_feat_ref

    def read_landmark(self,lm_path,t_path = 'org'):
        """
        @brief: Read and preprocess the dataset's landmarks.
        @param:
            ld_path: The path of the dataset's landmarks(xlsx file), and its format is [subject ID,X,Y,Z].
        @return:
            BMI_feat: Pretreated dataset's landmarks.
        """
        f_lm=pd.read_excel(lm_path)#Landmarks
        f_lm_id=f_lm['subject ID']
        f_lm_x = f_lm['X']
        f_lm_y = f_lm['Y']
        f_lm_z = f_lm['Z']

        f_lm_x_mean = f_lm_x.mean()
        f_lm_y_mean = f_lm_y.mean()
        f_lm_z_mean = f_lm_z.mean()
        feat_point = [] #Record landmarks of each face
        BMI_feat = [] #Landmarks Summary

        k_1 = np.ones(478)
        k_1 = k_1.reshape(478,-1)

        if len(self.temp) == 0:
            BMI_feat_ref = self.read_temp_landmark(t_path) #Template face for calibration
        else:
            BMI_feat_ref = self.temp

        for i in range(f_lm_id.shape[0]):
            if f_lm_id.iloc[i]!=478:
                f_lm_x_point = f_lm_x[i]-f_lm_x_mean
                f_lm_y_point = f_lm_y[i]-f_lm_y_mean
                f_lm_z_point = f_lm_z[i]-f_lm_z_mean
                if f_lm_id.iloc[i]==5:
                    f_lm_point_5 = (f_lm_x_point,f_lm_y_point,f_lm_z_point)
                if f_lm_id.iloc[i]==10:
                    f_lm_point_10 = (f_lm_x_point,f_lm_y_point,f_lm_z_point)
                
                pt_pos = [f_lm_x_point,f_lm_y_point,f_lm_z_point]
                feat_point.append(pt_pos)
            else : 
                f_lm_x_point = f_lm_x[i]-f_lm_x_mean
                f_lm_y_point = f_lm_y[i]-f_lm_y_mean
                f_lm_z_point = f_lm_z[i]-f_lm_z_mean
                pt_pos = [f_lm_x_point,f_lm_y_point,f_lm_z_point]
                feat_point.append(pt_pos)

                nose_dist = math.sqrt(math.pow((f_lm_point_5[0] - f_lm_point_10[0]), 2) 
                                    + math.pow((f_lm_point_5[1] - f_lm_point_10[1]), 2)
                                    + math.pow((f_lm_point_5[2] - f_lm_point_10[2]), 2))
                if np.array(BMI_feat).size == 0:
                    BMI_feat.append(feat_point)
                    BMI_feat = np.array(BMI_feat)
                    BMI_feat = BMI_feat/nose_dist
                    BMI_feat = BMI_feat.reshape(478,-1)
                    
                    BMI_feat = np.hstack((BMI_feat,k_1))
                    W_1 = np.linalg.pinv(BMI_feat).dot(BMI_feat_ref)
                    BMI_feat = BMI_feat.dot(W_1)
                    
                    BMI_feat = BMI_feat.reshape(1,1434)
                    feat_point = []
                else:
                    feat_point= np.array(feat_point)
                    feat_point = feat_point/nose_dist
                    feat_point = feat_point.reshape(478,-1)
                    
                    feat_point = np.hstack((feat_point,k_1))
                    W_1 = np.linalg.pinv(feat_point).dot(BMI_feat_ref)
                    feat_point = feat_point.dot(W_1)
                    
                    feat_point=feat_point.reshape(1,1434)
                    BMI_feat = np.vstack((BMI_feat,feat_point))
                    feat_point = []

        return BMI_feat
    
    def read_bmi(self,bmi_path,type='BMI'):
        """
        @brief: Read and preprocess the dataset's landmarks.
        @param:
            bmi_path: The path of the dataset's BMI and type(xlsx file), and its format is [image_index,BMI,type].
            type: 'BMI': The value of BMI, used for regression models.
                  'type': The type of BMI, used for classification models.
        @return:
            BMI_label: Dataset's BMI.
        """
        BMI_label = []
        f_bmi = pd.read_excel(bmi_path)#Dataset's BMI
        if type == 'BMI':
            f_bmi = f_bmi['BMI']#BMI
        elif type == 'type':
            f_bmi = f_bmi['type']#BMI

        for i in range(f_bmi.shape[0]):
            bmi = f_bmi[i]
            BMI_label.append(bmi)

        return BMI_label
    
    def get_image_x(self,imgpath,t_path = 'org'):
        BMI_feat_ref = self.read_temp_landmark(t_path)
        k_1 = np.ones(478)
        k_1 = k_1.reshape(478,-1)
        feature = []
        fl = FL.create_faciallandmark()
        coords = fl.get_landmarks(imgpath)
        if len(coords) != 0:
            for i in coords:
                feature.append([i.x,i.y,i.z])
            feature = np.array(feature)
            feature[:,0] = feature[:,0]-feature[:,0].mean()
            feature[:,1] = feature[:,1]-feature[:,1].mean()
            feature[:,2] = feature[:,2]-feature[:,2].mean()
            nose_dist = math.sqrt(math.pow((feature[4][0] - feature[9][0]), 2) 
                                    + math.pow((feature[4][1] - feature[9][1]), 2)
                                    + math.pow((feature[4][2] - feature[9][2]), 2))
            feature = feature/nose_dist
            feature = feature.reshape(478,-1)
            
            feature = np.hstack((feature,k_1))
            W_1 = np.linalg.pinv(feature).dot(BMI_feat_ref)
            feature = feature.dot(W_1)
            feature= feature.reshape(1,1434)
            return feature
        else:
            return []