import numpy as np
import os.path as path
from Facial_landmark import read_data as RD
from scipy.stats import pearsonr
from sklearn.datasets import make_friedman2
from sklearn import svm
from sklearn.model_selection import train_test_split
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

arr=[]
brr=[]
crr=[]
rd = RD.create_readdata()
two_up =  path.abspath(path.join(__file__ ,"../../../data"))
# print(two_up)
featpath = (two_up+ "/"+'VIP_G_lm.xlsx' )
bmipath = (two_up + "/" +'VIP_G_BMI.xlsx')
BMI_feat = rd.read_landmark(lm_path=featpath)
print(BMI_feat[0])
# BMI_label = rd.read_bmi(bmi_path=bmipath)

# x,y = BMI_feat,BMI_label
# # for i in range(100):
# X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size = 0.1)

# regr = svm.SVR(kernel='linear')
# # Fit to data using Maximum Likelihood Estimation of the parameters
# regr.fit(X_train, Y_train)
# joblib.dump(regr,'SVR_lm_BMI.pkl') 
# BMI_pred = regr.predict(X_test)
# r = pearsonr(Y_test,BMI_pred)
# MAE = np.mean(abs(Y_test - BMI_pred))
# #print('R =',r)
# b=r[0]
# c = r[1]
# arr.append(b)
# brr.append(MAE)
# crr.append(c)
# print('arr',arr)
# print('brr',brr)
# print('crr',crr)

#     if i%10 == 0:
#         print("finished:",i)
# arr_mean = np.mean(arr)
# arr_var = np.var(arr)
# crr_mean = np.mean(crr)
# crr_var = np.var(crr)
# brr_mean = np.mean(brr)
# brr_var = np.var(brr)
# print('Average value of R：',arr_mean)
# print('Standard deviation of R：',arr_var)
# print('Average value of MAE：',brr_mean)
# print('Standard deviation of MAE：',brr_var)
# print('Average value of P：',crr_mean)
# print('Standard deviation of P：',crr_var)