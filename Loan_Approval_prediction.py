# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 10:43:31 2019

@author: Sanjay-Sir
"""

#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

dataset = pd.read_csv("F:/My loan approval model/output.csv")


#sklearn requires all inputs to be numeric, we should convert all our categorical variables 
#into numeric by encoding the categories. This can be done using the following code:
from sklearn.preprocessing import LabelEncoder
var_mod = ['Loan_ID','Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    dataset[i] = le.fit_transform(dataset[i])
    

#Lets take a peek at the data
print(dataset.head(20))

#Splitting the Data set
X =dataset.iloc[:, 0:12].values
y =dataset.iloc[:, 12].values

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=7)


#Evaluating the model and training the Model
#Logistic Regression
modelLR = LogisticRegression()
modelLR.fit(x_train,y_train)
predictionsLR = modelLR.predict(x_test)
print(accuracy_score(y_test, predictionsLR))

#accuracy = accuracy_score(predictionsLR,y_test)
#print("Accuracy : %s" % "{0:.3%}".format(accuracy))

#Applying K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuraciesLR = cross_val_score(estimator = modelLR,X = x_train,y=y_train, cv = 10)
accuraciesLR.mean()

#writing the output for the Logistics regression
final_ID_LR = x_test[:,0] #taking the final id which is the id we created to verify the loan status by id

#Creating pandas dataframe from numpy array
final_logistics = pd.DataFrame({'ID_Number':final_ID_LR,'Loan_Status':predictionsLR})

final_logistics.to_csv('F:/My loan approval model/final_logistics.csv',index=False)

#Decision Tree
modelDT = DecisionTreeClassifier()
modelDT.fit(x_train,y_train)
predictionsDT = modelDT.predict(x_test)
print(accuracy_score(y_test, predictionsDT))


#accuracy = accuracy_score(predictionsDT,y_test)
#print("Accuracy : %s" % "{0:.3%}".format(accuracy))

#Applying K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuraciesLR = cross_val_score(estimator = modelDT,X = x_train,y=y_train, cv = 10)
accuraciesLR.mean()

#writing the output for the decision tree classifier
final_ID_DT = x_test[:,0] #taking the final id which is the id we created to verify the loan status by id

#Creating pandas dataframe from numpy array
final_decisiontree = pd.DataFrame({'ID_Number':final_ID_DT,'Loan_Status':predictionsDT})

final_decisiontree.to_csv('F:/My loan approval model/final_decisiontree.csv',index=False)


#RandomForest
modelRF = RandomForestClassifier(n_estimators=100) 
modelRF.fit(x_train,y_train)
predictionsRF = modelRF.predict(x_test)
print(accuracy_score(y_test,predictionsRF))

#accuracy = accuracy_score(predictionsRF,y_test)
#print("Accuracy : %s" % "{0:.3%}".format(accuracy))

#Applying K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuraciesLR = cross_val_score(estimator = modelRF,X = x_train,y=y_train, cv = 10)
accuraciesLR.mean()


#writing the output for the random Forest classifier
final_ID_RF = x_test[:,0] #taking the final id which is the id we created to verify the loan status by id

#Creating pandas dataframe from numpy array
final_randomforest = pd.DataFrame({'ID_Number':final_ID_RF,'Loan_Status':predictionsRF})

final_randomforest.to_csv('F:/My loan approval model/final_randomforest.csv',index=False)
