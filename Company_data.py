# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 21:15:16 2023

@author: Rahul
"""

import pandas as pd
import numpy as np

df=pd.read_csv("D:\\DS\\books\\ASSIGNMENTS\\Random Forests\\Company_Data.csv")
df.head()
df.info()
df.duplicated()
df[df.duplicated()]
df.isnull().any()
df.dtypes
df.corr()
df.corr().to_csv("Dtree.csv")

import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(data=df, hue = 'ShelveLoc')

# Boxplots
df.boxplot("CompPrice",vert=False)
Q1=np.percentile(df["CompPrice"],25)
Q3=np.percentile(df["CompPrice"],75)
IQR=Q3-Q1
LW=Q1-(1.5*IQR)
UW=Q3+(1.5*IQR)
df[df["CompPrice"]<LW].shape
df[df["CompPrice"]>UW].shape
df["CompPrice"]=np.where(df["CompPrice"]>UW,UW,np.where(df["CompPrice"]<LW,LW,df["CompPrice"]))

df.boxplot("Income",vert=False)
Q1=np.percentile(df["Income"],25)
Q3=np.percentile(df["Income"],75)
IQR=Q3-Q1
LW=Q1-(1.5*IQR)
UW=Q3+(1.5*IQR)
df[df["Income"]<LW].shape
df[df["Income"]>UW].shape
df["Income"]=np.where(df["Income"]>UW,UW,np.where(df["Income"]<LW,LW,df["Income"]))

df.boxplot("Advertising",vert=False)
df.boxplot("Population",vert=False)

df.boxplot("Price",vert=False)
Q1=np.percentile(df["Price"],25)
Q3=np.percentile(df["Price"],75)
IQR=Q3-Q1
LW=Q1-(1.5*IQR)
UW=Q3+(1.5*IQR)
df[df["Price"]<LW].shape
df[df["Price"]>UW].shape
df["Price"]=np.where(df["Price"]>UW,UW,np.where(df["Price"]<LW,LW,df["Price"]))

df.boxplot("Age",vert=False)
df.boxplot("Education",vert=False)

# Coverting Sales into categorical
df["Sales"] = pd.cut(df["Sales"], bins=[0,4.2,8.01,12.01,16.27],labels=["poor","good","very good","excellent"])
df

# Splitting the variables
Y=df["Sales"]
X=df.iloc[:,1:]
X.dtypes

# Standardization
from sklearn.preprocessing import MinMaxScaler
MM=MinMaxScaler()
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()

X["CompPrice"]=MM.fit_transform(X[["CompPrice"]])

X["Income"]=MM.fit_transform(X[["Income"]])

X["Advertising"]=MM.fit_transform(X[["Advertising"]])

X["Population"]=MM.fit_transform(X[["Population"]])

X["Price"]=MM.fit_transform(X[["Price"]])

X["Age"]=MM.fit_transform(X[["Age"]])

X["Education"]=MM.fit_transform(X[["Education"]])

X["ShelveLoc"]=LE.fit_transform(X["ShelveLoc"])
X["ShelveLoc"]=pd.DataFrame(X["ShelveLoc"])

X["Urban"]=LE.fit_transform(X["Urban"])
X["Urban"]=pd.DataFrame(X["Urban"])

X["US"]=LE.fit_transform(X["US"])
X["US"]=pd.DataFrame(X["US"])
X

Y=LE.fit_transform(df["Sales"])
Y=pd.DataFrame(Y)

# Train and Test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4)

# Model fitting
from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier(max_depth=6,max_leaf_nodes=20)
RF.fit(X_train,Y_train)
Y_predtrain=RF.predict(X_train)
Y_predtest=RF.predict(X_test)

from sklearn.metrics import accuracy_score
ac1=accuracy_score(Y_train,Y_predtrain)
print("accuracy_score1: ", ac1)

ac2=accuracy_score(Y_test,Y_predtest)
print("accuracy_score2: ", ac2)

# Bagging
from sklearn.ensemble import BaggingClassifier
RF=RandomForestClassifier(max_depth=5)
Bag=BaggingClassifier(base_estimator=RF,max_samples=0.6,n_estimators=100)
Bag.fit(X_train,Y_train)                     
Y_predtrain=Bag.predict(X_train)
Y_predtest=Bag.predict(X_test)

from sklearn.metrics import accuracy_score
ac1=accuracy_score(Y_train,Y_predtrain)
print("accuracy_score1: ", ac1)

ac2=accuracy_score(Y_test,Y_predtest)
print("accuracy_score1: ", ac2)

# Entropy method
Training_accuracy = []
Test_accuracy = []

for i in range(1,12):
    regressor = RandomForestClassifier(max_depth=i,criterion="entropy") 
    regressor.fit(X_train,Y_train)
    Y_pred_train = regressor.predict(X_train)
    Y_pred_test = regressor.predict(X_test)
    Training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    Test_accuracy.append(accuracy_score(Y_test,Y_pred_test))

pd.DataFrame(Training_accuracy)
pd.DataFrame(Test_accuracy)
pd.concat([pd.DataFrame(range(1,12)) ,pd.DataFrame(Training_accuracy),pd.DataFrame(Test_accuracy)],axis=1)    

# Model fitting
from sklearn.ensemble import RandomForestRegressor
RF=RandomForestRegressor(max_depth=7,max_leaf_nodes=20)
RF.fit(X_train,Y_train)
Y_predtrain=RF.predict(X_train)
Y_predtest=RF.predict(X_test)

# r2_score
from sklearn.metrics import r2_score
rs1=r2_score(Y_train,Y_predtrain)
print("r2_score1: ", rs1)
rs2=r2_score(Y_test,Y_predtest)
print("r2_score2: ", rs2)

# Bagging
from sklearn.ensemble import BaggingRegressor
RF=RandomForestRegressor(max_depth=7)
Bag=BaggingRegressor(base_estimator=RF,max_samples=0.6,n_estimators=100)
Bag.fit(X_train,Y_train)                     
Y_predtrain=Bag.predict(X_train)
Y_predtest=Bag.predict(X_test)

from sklearn.metrics import r2_score
rs1=r2_score(Y_train,Y_predtrain)
print("r2_score1: ", rs1)
rs2=r2_score(Y_test,Y_predtest)
print("r2_score1: ", rs2)



















































