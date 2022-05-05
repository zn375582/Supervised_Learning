#support vector machine implementation for comparison purposes
#source: https://scikit-learn.org/stable/modules/svm.html
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import sklearn.model_selection as skm
from sklearn import svm
from sklearn.metrics import accuracy_score

#load data into numpy array
data_x = np.genfromtxt("cleveland_heart_dis_xval_processed.csv", delimiter=",", skip_header=1)
data_y = np.genfromtxt("cleveland_heart_dis_yval.csv", delimiter=",", skip_header=1)

#If y is 1,2,3 then they have heart disease, if 0 then don't
for index, value in enumerate(data_y):
    if value > 1:
        data_y[index] = 1


#remove 20% of the examples and keep them for testing
xtrain, xtest, ytrain, ytest = skm.train_test_split(data_x, data_y, test_size=.2, random_state=34)

#Split the remaining examples into training (75%) and validation (25%)
x_train, x_valid, y_train, y_valid = skm.train_test_split(xtrain, ytrain, test_size=.25, random_state=34)

#svm from sklearn library
clf = svm.SVC(kernel='linear', C=1).fit(xtrain, ytrain)
clf_prediction = clf.predict(xtest)
print(accuracy_score(ytest, clf_prediction)*100)

#Novelty: predict missing attributes

#80.327% accurate


