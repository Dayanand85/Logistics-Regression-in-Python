# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 12:36:43 2022

@author: Dayanand

"""

# loading library

import os
import seaborn as sns 
import pandas as pd
import numpy as np
from matplotlib.pyplot import figure
from matplotlib.backends.backend_pdf import PdfPages

#setting directory path
os.getcwd()
os.chdir("C:\\Users\\tk\\Desktop\\DataScience\\Analytics Vidhya\\DataSet\\Loan Amount")

# loading files

rawData=pd.read_csv("train.csv")
predictionData=pd.read_csv("test.csv")

rawData.shape
predictionData.shape

rawData.columns
predictionData.columns

# Adding Loan_Status column in predictionData
predictionData["Loan_Status"]=0
predictionData.shape

# sampling rawData into trainData & testData
from sklearn.model_selection import train_test_split
trainData,testData=train_test_split(rawData,train_size=0.8,random_state=2410)
trainData.shape
testData.shape

# Adding source column in trainData,testData & predictionData

trainData["Source"]="Train"
testData["Source"]="Test"

predictionData["Source"]="Prediction"


# combine train,test & prediction Datasets

fullData=pd.concat([trainData,testData,predictionData],axis=0)
fullData.shape

# Event rate

fullData.loc[fullData["Source"]=="Train","Loan_Status"].value_counts()/trainData["Loan_Status"].shape[0]

# 1's class-68%
# 0's class-32%

# removing identifier column

fullData.columns
fullData.drop(["Loan_ID"],axis=1,inplace=True)
fullData.shape

############## Univariate Analysis

# missing value check

fullData.isna().sum()

# We have missing values in Gender,Dependants,Self_Employed,
# LaonAmount,Loan_Amount_Term,Credit_History,

# Gender

fullData["Gender"].dtypes
tempMode=fullData.loc[fullData["Source"]=="Train","Gender"].mode()[0]
tempMode
fullData["Gender"].fillna(tempMode,inplace=True)
fullData["Gender"].isna().sum()

# Dependents

fullData["Dependents"].dtypes
tempMode=fullData.loc[fullData["Source"]=="Train","Dependents"].mode()[0]
tempMode
fullData["Dependents"].fillna(tempMode,inplace=True)
fullData["Dependents"].isna().sum()

# Self_Employed

fullData["Self_Employed"].dtypes
tempMode=fullData.loc[fullData["Source"]=="Train","Self_Employed"].mode()[0]
tempMode
fullData["Self_Employed"].fillna(tempMode,inplace=True)
fullData["Self_Employed"].isna().sum()

# LoanAmount

fullData["LoanAmount"].dtypes
tempMedian=fullData.loc[fullData["Source"]=="Train","LoanAmount"].median()
tempMedian
fullData["LoanAmount"].fillna(tempMedian,inplace=True)
fullData["LoanAmount"].isna().sum()

# Loan_Amount_Term

fullData["Loan_Amount_Term"].dtypes
tempMedian=fullData.loc[fullData["Source"]=="Train","Loan_Amount_Term"].median()
tempMedian
fullData["Loan_Amount_Term"].fillna(tempMedian,inplace=True)
fullData.isna().sum()

# Credit_History

fullData["Credit_History"].dtypes
fullData["Credit_History"].value_counts()
tempMedian=fullData.loc[fullData["Source"]=="Train","Credit_History"].median()
tempMedian
fullData["Credit_History"].fillna(tempMedian,inplace=True)
fullData["Credit_History"].isna().sum()

# summary of datasets

fullData_Summary=fullData.describe()

# Levels change of category column as per data description
# Dependants

fullData["Dependents"].dtypes
fullData["Dependents"].value_counts()
fullData["Dependents"].replace({"0":"No_Depen","1":"One_Depen","2":"Two_Depen","3+":"Three_Depen"},inplace=True)
fullData["Dependents"].value_counts()

# Credit_History

fullData["Credit_History"].dtypes
fullData["Credit_History"].value_counts()
fullData["Credit_History"].replace({0:"Bad",1:"Good"},inplace=True)
fullData["Credit_History"].value_counts()

# Bivariate Analysis
# continuous indep variables and category depen var

trainDf=fullData.loc[fullData["Source"]=="Train"]
continuousvars=trainDf.columns[trainDf.dtypes!=object]
continuousvars

filename="continuousvars.pdf"
pdf=PdfPages(filename)
for colNumber,colName in enumerate(continuousvars):
    figure()
    sns.boxplot(y=trainDf[colName],x=trainDf["Loan_Status"])
    pdf.savefig(colNumber+1)
pdf.close()

# Bivariate Analysis
#categorical independent Variable and category dependent variable
#categ_vars=trainDf.columns.drop(continuousvars)

categoricalvars=trainDf.columns[trainDf.dtypes==object]
categoricalvars
filename="categoricalvars.pdf"

pdf=PdfPages(filename)

for colNumber,colName in enumerate(categoricalvars):
    print(colNumber,colName)
    figure()
    sns.histplot(trainDf,x=colName,hue="Loan_Status",stat="probability",multiple="fill")
    #sns.histplot(trainDf, x="Education", hue="Loan_Status", stat="probability", multiple="fill")
    #sns.distplot(trainDf,x=colName,hue="Loan_Status",multiple="stack")
    pdf.savefig(colNumber+1)

pdf.close()

# Boxplot 

fullData.describe()
fullData.columns
VarForOutlier=["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"]
for colName in VarForOutlier:
    print(colName)
    
    Q1=np.percentile(fullData.loc[fullData["Source"]=="Train",colName],25)
    
    Q3=np.percentile(fullData.loc[fullData["Source"]=="Train",colName],75)
    IQR=Q3-Q1
    LB=Q1-1.5*IQR
    UB=Q3+1.5*IQR
    fullData[colName]=np.where(fullData[colName]<LB,LB,fullData[colName])
    fullData[colName]=np.where(fullData[colName]>UB,UB,fullData[colName])

# change Dependent variable category manualy

fullData["Loan_Status"].value_counts()
fullData["Loan_Status"]=np.where(fullData["Loan_Status"]=="Y",1,0)
fullData["Loan_Status"].value_counts()

# Dummy variable creation

fullData2=pd.get_dummies(fullData,drop_first=True)
fullData2.shape


# divide data into train,test & prediction

train=fullData2[fullData2["Source_Train"]==1].drop(["Source_Train","Source_Test"],axis=1)
train.shape
test=fullData2[fullData2["Source_Test"]==1].drop(["Source_Train","Source_Test"],axis=1)
test.shape
prediction=fullData2[(fullData2["Source_Train"]==0) & (fullData2["Source_Test"]==0)].drop(["Source_Train","Source_Test"],axis=1)
prediction.shape

# Divide independet & dependent columns

trainX=train.drop(["Loan_Status"],axis=1).copy()
trainY=train["Loan_Status"].copy()
trainX.shape
trainY.shape
testX=test.drop(["Loan_Status"],axis=1).copy()
testY=test["Loan_Status"]
predictionX=prediction.drop(["Loan_Status"],axis=1).copy()
testX.shape
testY.shape
predictionX.shape

# Add intercept columns

from statsmodels.api import add_constant
trainX=add_constant(trainX)
trainX.shape
testX=add_constant(testX)
testX.shape
predictionX=add_constant(predictionX)
predictionX.shape

#VIF check

from statsmodels.stats.outliers_influence import variance_inflation_factor
tempmaxVIF=10
maxVIF=10
trainXcopy=trainX.copy()
highVIFColumn=[]
counter=1

while(tempmaxVIF>=maxVIF):
    #print(counter)
    
    # create an empty tempDf to store VIF
    tempDf=pd.DataFrame()
    
    # calculate VIF using list comprehension
    tempDf["VIF"]=[variance_inflation_factor(trainXcopy.values,i) for i in range(trainXcopy.shape[1])]
    
    #create a new column to store column Names
    tempDf["ColName"]=trainXcopy.columns
    # drop NA columns if any calculation mistake is done
    tempDf.dropna(inplace=True)
    # sort the VIF and store the highest VIF values columns
    tempColumnName=tempDf.sort_values(["VIF"],ascending=False).iloc[0,1]
    # sort the VIF and store in VIF columns
    tempmaxVIF=tempDf.sort_values(["VIF"],ascending=False).iloc[0,0]
    
    
    if(tempmaxVIF>=maxVIF):
        trainXcopy.drop(tempColumnName,axis=1)
        highVIFColumn.append(tempColumnName)
        #print(tempColumnName)
    
    counter=counter+1

#highVIFColumn

# We have got only one column Loan_Amount_Term which has got high VIF

trainX=trainX.drop("Loan_Amount_Term",axis=1)
testX=testX.drop("Loan_Amount_Term",axis=1)
predictionX=predictionX.drop("Loan_Amount_Term",axis=1)

trainX.shape
testX.shape
predictionX.shape

# Build Logistics Regresstion Model using statsmodels.api

from statsmodels.api import Logit
Model_Def=Logit(trainY,trainX)
M1=Model_Def.fit()
M1.summary()

# Using random forest algorithm to select significant variable

from sklearn.ensemble import RandomForestClassifier
M2_RF=RandomForestClassifier(random_state=2410).fit(trainX,trainY)

sing_var=pd.DataFrame()
sing_var["ImpVar"]=M2_RF.feature_importances_
sing_var["colName"]=trainX.columns

tempMedian=sing_var["ImpVar"].median()
tempColName=[]
for i in range(sing_var.shape[0]):
    print(i)
    if(sing_var["ImpVar"][i]<=tempMedian):
        tempColName.append(sing_var["colName"][i])
    
trainX=trainX.drop(tempColName,axis=1)    
trainX.shape

testX=testX.drop(tempColName,axis=1)
testX.shape

predictionX=predictionX.drop(tempColName,axis=1)
predictionX.shape

# Model building on new data sets
F_M=Logit(trainY,trainX).fit()
F_M.summary()

# prediction on test data
Test_Prob=F_M.predict(testX)

# converting probabilities into class
Test_Class=np.where(Test_Prob>=0.5,1,0)

# confusion matrix
conf_mat=pd.crosstab(Test_Class,testY)
conf_mat
#Accuracy
from sklearn.metrics import classification_report
print(classification_report(testY,Test_Class))#Actual,prediction

# AUC,ROC curve

from sklearn.metrics import roc_curve,auc
Train_Predict=F_M.predict(trainX)

# fpr,tpr & cutoff threshhold
fpr,tpr,cut_off=roc_curve(trainY,Train_Predict)

# plot roc_curve
import seaborn as sns
sns.lineplot(fpr,tpr)

# area under curve
auc(fpr,tpr)
#75%

# selecting new cut off points
Cut_Off_Table=pd.DataFrame()
Cut_Off_Table["FPR"]=fpr
Cut_Off_Table["TPR"]=tpr
Cut_Off_Table["Cut_Off"]=cut_off

Cut_Off_Table["Diff"]=tpr-fpr

# new cutoff=0.54
Test_Class1=np.where(Test_Prob>=0.54,1,0)

#print classification report
print(classification_report(testY,Test_Class1))

# prediction on prediction datasets

prediction_prob=F_M.predict(predictionX)
Output_File=pd.DataFrame()
Output_File["Loan_ID"]=predictionData["Loan_ID"]
Output_File["Loan_Status"]=np.where(prediction_prob>=0.54,1,0)
Output_File["Loan_Status"].value_counts()
Output_File["Loan_Status"].replace({0:"N",1:"Y"},inplace=True)
Output_File.to_csv("OutPutFile.csv")
