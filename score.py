import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.svm import SVC
import sklearn.datasets as skdata
import pandas as pd
import sklearn.metrics
from PIL import Image
from scipy.misc import imread
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix


y_train= np.zeros(100)

imagenes= []
for i in range (100):
    if(i%2==1):
        y_train[i]=1
    img = imread('train/'+str(i+1)+'.jpg')[0]
    imagenes.append(img.flatten())


test =[]
files_test = glob.glob("test/*.jpg")
files_test =files_test
n_test = len (files_test)
for i in range(n_test):
    img = imread(files_test[i])[0]
    test.append(img.flatten())

x_train = imagenes
x_test = test

y_test = np.array(pd.read_csv('test/truth_test.csv')['Target'])

scaler = StandardScaler()
x_train = scaler.fit_transform(imagenes)
x_test=scaler.transform(x_test)


cov = np.cov(x_train.T)
valores, vectores = np.linalg.eig(cov)

valores = np.real(valores)
vectores = np.real(vectores)

ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]

train_trans = np.dot(x_train, vectores)
test_trans = np.dot(x_test,vectores)
train_trans[:,:100]
svclassifier = SVC(kernel='rbf', C=10)
svclassifier.fit(x_train, y_train)
y_predict=svclassifier.predict(x_test)

print(classification_report(y_predict,y_test))
out= open("test/predict_test.csv","w")
out.write("Name,Target\n")
for f,p in zip(files_test, y_predict):
    print (f.split("/")[-1], p)
    out.write("{},{}\n".format(f.split("/")[-1],p))
    
out.close()