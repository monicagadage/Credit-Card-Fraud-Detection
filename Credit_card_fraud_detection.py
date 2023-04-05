#!/usr/bin/env python
# coding: utf-8

# In[62]:


# EDA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import sklearn as sk
import pprint
import plotly.express as px
import datetime as dt
from sklearn.metrics import roc_auc_score as roc
# Math
from scipy.stats import loguniform
# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# Metrics
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score, classification_report, matthews_corrcoef
from sklearn.metrics import mean_absolute_error
# Scaling
from sklearn.preprocessing import StandardScaler
# Processing Data
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score, RandomizedSearchCV, PredefinedSplit
from sklearn.model_selection import PredefinedSplit
# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
# Neural Nets
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
# Counter
from collections import Counter
from collections import defaultdict

# Resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from imblearn.pipeline import Pipeline
df_metrics_table = pd.DataFrame()
# Save
import joblib
import pickle

import tensorflow as tf
from tensorflow import keras

import seaborn as sns
sns.set(rc={'figure.figsize':(11,6)})

eu_data =  pd.read_csv('/Users/monikagadage/Desktop/csci693/creditcard.csv')
syn_train_data = pd.read_csv('/Users/monikagadage/Desktop/csci693/fraudTrain.csv')
syn_test_data = pd.read_csv('/Users/monikagadage/Desktop/csci693/fraudTest.csv')


# In[4]:


eu_data.head(3)


# In[5]:


syn_train_data.head(3)


# In[6]:


print(f' Total rows and columns: \n {eu_data.shape}')


# In[7]:


print(f' Total rows and columns: \n {syn_train_data.shape}')


# In[8]:


eu_data.isna().sum()


# In[9]:


# check for the balance of data
print(f' Fraud vs. Non Fraud: \n {np.round(eu_data["Class"].value_counts().sort_values() / len(eu_data) * 100, 2)}')


# In[10]:


# check for the balance of data
print(f' Fraud vs. Non Fraud: \n {np.round(syn_train_data["is_fraud"].value_counts().sort_values() / len(syn_train_data) * 100, 2)}')


# In[15]:


# using groupby to calculate median amounts for each class
print(eu_data.groupby('Class')['Amount'].median())
eu_data.groupby('Class')['Amount'].median().plot.bar()
ax = plt.gca()
ax.tick_params(axis='x', labelrotation = 0)
plt.title('Median Credit Card Purchase Amount for Non Fruad vs. Fraud')


# In[16]:


# using groupby to calculate median amounts for each class
print(syn_train_data.groupby('is_fraud')['amt'].median())
syn_train_data.groupby('is_fraud')['amt'].median().plot.bar()
ax = plt.gca()
ax.tick_params(axis='x', labelrotation = 0)
plt.title('Median Credit Card Purchase Amount for Non Fruad vs. Fraud')


# In[17]:


# plot all credit card purchase amounts
eu_data['Amount'].plot(figsize = (14, 4 ))
plt.title("All Credit Card Purchases in the Dataset")
plt.xlabel("Individual Purchases")
plt.ylabel("Amount")
# noticeable outliers


# In[18]:


# plot all credit card purchase amounts
syn_train_data['amt'].plot(figsize = (14, 4 ))
plt.title("All Credit Card Purchases in the Dataset")
plt.xlabel("Individual Purchases")
plt.ylabel("Amount")
# noticeable outliers


# In[35]:


# heatmap to illustrate correlation between features 
ax = plt.axes()
sns.heatmap(eu_data.corr(),ax=ax, cmap='YlGnBu', annot=True, fmt='.2f')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
sns.set(rc = {'figure.figsize':(18,9)})
# plt.figure(figsize = (18, 9))
ax.set_title('Linear Correlation Heatmap Between Features in the Dataset', fontsize= 25, pad=20)


# In[19]:


#Plotting the heat map to find the correlation between the columns
sns.heatmap(syn_train_data.corr(),annot=True, linewidths=0.5, cmap = "Blues")


# In[23]:


#Calculating age from the transaction time and the date of birth columns
syn_train_data["age"] = pd.DatetimeIndex(syn_train_data["trans_date_trans_time"]).year-pd.DatetimeIndex(syn_train_data["dob"]).year
#Defining a bucket to categorize age into different age groups
def age_group(row):
    rows = row["age"]
    if rows <20:
        return "less than 20"
    elif rows >=20 and rows<30:
        return "20 to 30"
    elif rows >=30 and rows<40:
        return "30 to 40"
    elif rows >=40 and rows<50:
        return "40 to 50"
    elif rows >=50 and rows<60:
        return "50 to 60"
    elif rows >=60 and rows<70:
        return "60 to 70"
    elif rows >=70 and rows<80:
        return "70 to 80"
    elif rows >=80 and rows<90:
        return "80 to 90"
    else:
        return "greater than 90"
syn_train_data["age_group"] = syn_train_data.apply(age_group,axis=1)


# In[25]:


#Dropping age and date of birth from the dataframe
syn_train_data = syn_train_data.drop(['age','dob'],1)


# In[27]:


#Plotting fraud transactions for different age groups
order = ["less than 20","20 to 30","30 to 40","40 to 50","50 to 60","60 to 70","70 to 80","80 to 90","greater than 90"]
sns.countplot(x="age_group",data=syn_train_data[syn_train_data["is_fraud"]==1], order = order)


# In[28]:


#Plotting fraud transactions with respect to gender
sns.countplot(x="gender",data=syn_train_data[syn_train_data["is_fraud"]==1])


# In[29]:


#Plotting fraud transactions with respect to states
sns.countplot(x="state",data=syn_train_data[syn_train_data["is_fraud"]==1])


# In[31]:


syn_train_data["hour"] = pd.DatetimeIndex(syn_train_data["trans_date_trans_time"]).hour
#Plotting fraud transactions with respect to the hour of day
sns.countplot(x="hour",data=syn_train_data[syn_train_data["is_fraud"]==1])
#It can be concluded that most of the fraud transactions happened during midnight


# In[32]:


fig, ax = plt.subplots(1, 2, figsize=(18,4))

Val_Amount = eu_data['Amount'].values
Val_Time = eu_data['Time'].values

sns.distplot(Val_Amount, ax=ax[0], color='r')
ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
ax[0].set_xlim([min(Val_Amount), max(Val_Amount)])

sns.distplot(Val_Time, ax=ax[1], color='b')
ax[1].set_title('Distribution of Transaction Time', fontsize=14)
ax[1].set_xlim([min(Val_Time), max(Val_Time)])

plt.show()


# In[33]:


plt.figure(figsize=(8,4))
Nofraud_data = eu_data [eu_data['Class']==0]
plt.figure(figsize=(12,4), dpi=80)
sns.distplot(Nofraud_data [ 'Time']/60/60, bins=48)
plt.xticks(np.arange(0,54,6))
plt.xlim([0,48])
plt.xlabel('Time After First Transaction (hr)',fontsize=13)
plt.ylabel('Number of transactions',fontsize=13)
plt.title('Transaction Times for genuine users ',fontsize=15)


# In[34]:


plt.figure(figsize=(8,4))
Fraud_data = eu_data [eu_data['Class']==1]
plt.figure(figsize=(12,4), dpi=80)
sns.distplot(Fraud_data[ 'Time']/60/60, bins=48)
plt.xticks(np.arange(0,54,6))
plt.xlim([0,48])
plt.xlabel('Time After First Transaction (hr)',fontsize=13)
plt.ylabel('Number of transactions',fontsize=13)
plt.title('Transaction Times for Non-genuine (fraudulent) users',fontsize=15)


# In[36]:


# Creating fraudulent dataframe
data_fraud = eu_data[eu_data['Class'] == 1]
# Creating non fraudulent dataframe
data_non_fraud = eu_data[eu_data['Class'] == 0]
# Distribution plot
plt.figure(figsize=(8,5))
ax = sns.distplot(data_fraud['Time'],label='fraudulent',hist=False)
ax = sns.distplot(data_non_fraud['Time'],label='non fraudulent',hist=False)
ax.set(xlabel='Seconds elapsed between the transction and the first transction')
plt.show()


# In[37]:


# Dropping the Time column
eu_data.drop('Time', axis=1, inplace=True)


# In[38]:


# Distribution plot
plt.figure(figsize=(8,5))
ax = sns.distplot(data_fraud['Amount'],label='fraudulent',hist=False)
ax = sns.distplot(data_non_fraud['Time'],label='non fraudulent',hist=False)
ax.set(xlabel='Transction Amount')
plt.show()


# In[40]:


syn_train_data.drop_duplicates(inplace=True)
syn_train_data['age']=dt.date.today().year-pd.to_datetime(synthetic_train_data['dob']).dt.year

#subset the training data to include only the features that we need
syn_train_data=syn_train_data[['category','amt','gender','age','is_fraud']]


# In[41]:


#convert category to dummy variables
syn_train_data=pd.get_dummies(syn_train_data, drop_first=True)
y_train_synthetic=syn_train_data['is_fraud'].values
X_train_synthetic=syn_train_data.drop("is_fraud", axis='columns').values
y_train_syn=syn_train_data['is_fraud']
X_train_syn=syn_train_data.drop("is_fraud", axis='columns')


# In[42]:


syn_test_data.drop_duplicates(inplace=True)
syn_test_data['age']=dt.date.today().year-pd.to_datetime(syn_test_data['dob']).dt.year
test=syn_test_data[['category','amt','gender','age','is_fraud']]
#convert category to dummy variables
test=pd.get_dummies(test, drop_first=True)
y_test_synthetic=test['is_fraud'].values
X_test_synthetic=test.drop("is_fraud", axis='columns').values
y_test_syn=test['is_fraud']
X_test_syn=test.drop("is_fraud", axis='columns')


# In[43]:


#seperating positive and negative classes
europe_positiveDataset = eu_data.loc[eu_data['Class'] == 1]
europe_negativeDataset = eu_data.loc[eu_data['Class'] == 0]
#creating training and testing set with negative class split 1:1 and positive class split 4:1, also keeping random_state constant so that all splits are same
positiveTrain, positiveTest = train_test_split(europe_positiveDataset, test_size=0.2, random_state=21)
negativeTrain, negativeTest = train_test_split(europe_negativeDataset, test_size=0.5)
trainDataset = positiveTrain.append(negativeTrain)
testDataset = positiveTest.append(negativeTest)
y_train_eu = trainDataset['Class']
y_test_eu = testDataset['Class']
X_train_eu = trainDataset.drop(columns=['Class'])
X_test_eu = testDataset.drop(columns=['Class'])


# In[56]:


#Defining a generic function to implement models and returning data values
def model_implementation(model,X_train,X_test,y_train,y_test):
    if model == "LogisticRegression":
        model = "Logistic Regression"
        model_impl = LogisticRegression()
    elif model == "DecisionTree":
        model = "Decision Tree"
        model_impl = DecisionTreeClassifier(random_state=0, max_depth=2)
    elif model == "RandomForestClassifier":
        model = "Random Forest Classifier"
        model_impl = RandomForestClassifier(random_state=0,max_depth= 10, max_features= 5,min_samples_leaf= 30, min_samples_split= 100, n_estimators= 500)
    elif model == "GaussianNaiveBias":
        model = "Gaussian Naive Bias"
        model_impl = GaussianNB()
    elif model == "LinearRegression":
        model = "LinearRegression"
        model_impl = LinearRegression()
        
    model_impl.fit(X_train,y_train)
    pred_train=model_impl.predict(X_train)
    pred_test=model_impl.predict(X_test)
    
    if model == "LinearRegression":
        for i in range(0, len(pred_train)):
            if(pred_train[i]>=0.5):
                pred_train[i]=1
            else:
                pred_train[i]=0
        for i in range(0, len(pred_test)):
            if(pred_test[i]>=0.5):
                pred_test[i]=1
            else:
                pred_test[i]=0


    accuracy_train = accuracy_score(pred_train,y_train)
    accuracy_test = accuracy_score(pred_test,y_test)
    confusionMatrix = confusion_matrix(y_test,pred_test)
    classificationReport = classification_report(y_test,pred_test)
    mae = mean_absolute_error(y_train,pred_train)
    roc_train = roc(y_train, pred_train)
    roc_test = roc(y_test, pred_test)
    f1_score = metrics.f1_score(y_test,pred_test)
    recall = metrics.recall_score(y_test, pred_test)
    print("Model Implemented: ", model)
    print("Accuracy on Training Set: ", accuracy_train)
    print('Accuracy on Validation Set: ', accuracy_test)
    print('ROC AUC Score Train: ',roc(y_train, pred_train))
    print('ROC AUC Score Test: ',roc(y_test, pred_test))
    print(f'(train) MCC: {metrics.matthews_corrcoef(y_train, pred_train)}')
    print(f'(train) F1: {metrics.f1_score(y_train,pred_train)}')
    print(f'(train) Recall: {metrics.recall_score(y_train, pred_train)}')
    print('Mean absolute error: ',mae)
    print('Confusion Matrix\n', confusionMatrix)
    print('Classification Report\n', classificationReport)
    fig, (ax1) = plt.subplots(1, figsize = (10, 4))
    sns.heatmap(confusionMatrix, annot=True,ax = ax1, fmt = 'g')
    ax1.set_xlabel('Predicted labels');ax1.set_ylabel('True labels') 
    ax1.set_title('Confusion Matrix before Fine tuning')
    ax1.xaxis.set_ticklabels(['Non Fraudulent', 'Fradulent']); ax1.yaxis.set_ticklabels(['Non Fraudulent', 'Fradulent'])
    

    return [accuracy_train,accuracy_test,roc_test,recall,f1_score]


# An ROC curve is a graph showing the performance of a classification model at all classification thresholds. This curve plots two parameters:
# True Positive Rate : recall
# False Positive Rate

# In[46]:


# function for saving metric data to df and displaying/returning the df

model_eval = {
    'model': '',
    'roc_test' : '',
    'recall_test' : '',
    'f1_score_test' : ''
}

def add_model_eval(model, roc_test, recall_test, f1_score_test):
    model_eval['model'] = model
    model_eval['roc_test'] =  f'{roc_test: .2f}'
    model_eval['recall_test'] = f'{recall_test: .2f}'
    model_eval['f1_score_test'] = f'{f1_score_test: .2f}'
    
 

def view_models_eval(df_results):
    
    
    df_results = df_results.append(model_eval, ignore_index=True)
    df_results = df_results.sort_values(by=['recall_test'], ascending=[False])
    # display
    # display(df_results.style.hide_index())

    return df_results


# In[57]:


LirEu = model_implementation("LinearRegression",X_train_eu,X_test_eu,y_train_eu,y_test_eu)


# In[63]:


# add Logistic Regression - Basline
add_model_eval('Linear Regression Eu Imbalance',LirEu[2], LirEu[3], LirEu[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[64]:


LirSyn= model_implementation("LinearRegression",X_train_syn,X_test_syn,y_train_syn,y_test_syn)


# In[65]:


# add Logistic Regression - Basline
add_model_eval('Linear Regression Syn Imbalance',LirSyn[2], LirSyn[3], LirSyn[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[66]:


#Gaussian Naive Bias
GuEu= model_implementation("GaussianNaiveBias",X_train_eu,X_test_eu,y_train_eu,y_test_eu)


# In[67]:


# add Logistic Regression - Basline
add_model_eval('GaussianNaiveBias Eu Imbalance',GuEu[2], GuEu[3], GuEu[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[68]:


GuSyn= model_implementation("GaussianNaiveBias",X_train_syn,X_test_syn,y_train_syn,y_test_syn)


# In[69]:


# add Logistic Regression - Basline
add_model_eval('GaussianNaiveBias Syn Imbalance',GuSyn[2], GuSyn[3], GuSyn[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[70]:


RfEu = model_implementation("RandomForestClassifier",X_train_eu,X_test_eu,y_train_eu,y_test_eu)


# In[71]:


# add Logistic Regression - Basline
add_model_eval('RandomForestClassifier Eu Imbalance',RfEu[2], RfEu[3], RfEu[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[72]:


RfSyn = model_implementation("RandomForestClassifier",X_train_syn,X_test_syn,y_train_syn,y_test_syn)


# In[73]:


# add Logistic Regression - Basline
add_model_eval('RandomForestClassifier Syn Imbalance',RfSyn[2], RfSyn[3], RfSyn[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[74]:


DtEu = model_implementation("DecisionTree",X_train_eu,X_test_eu,y_train_eu,y_test_eu)


# In[75]:


# add Logistic Regression - Basline
add_model_eval('DecisionTree Eu Imbalance',DtEu[2], DtEu[3], DtEu[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[76]:


DtSync = model_implementation("DecisionTree",X_train_syn,X_test_syn,y_train_syn,y_test_syn)


# In[77]:


# add Logistic Regression - Basline
add_model_eval('DecisionTree Syn Imbalance',DtSync[2], DtSync[3], DtSync[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[78]:


LrEu = model_implementation("LogisticRegression",X_train_eu,X_test_eu,y_train_eu,y_test_eu)


# In[79]:


# add Logistic Regression - Basline
add_model_eval('LogisticRegression Eu Imbalance',LrEu[2], LrEu[3], LrEu[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[80]:


LrSync = model_implementation("LogisticRegression",X_train_syn,X_test_syn,y_train_syn,y_test_syn)


# In[81]:


# add Logistic Regression - Basline
add_model_eval('LogisticRegression Syn Imbalance',LrSync[2], LrSync[3], LrSync[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[267]:


plot_data = pd.DataFrame({
   'Europe': [LirEu[0],GuEu[0],LrEu[0],DtEu[0],RfEu[0]],
   'Synthatic': [LirSyn[0],GuSyn[0],LrSync[0],DtSync[0],RfSyn[0]]
   }, index=["Lin Reg","GNB","Log Reg","DT","RF"])
plt.title("Imbalanced data Training Analysis")
plt.plot(plot_data)


# In[268]:


plot_data_test = pd.DataFrame({
   'Europe': [LirEu[1],GuEu[1],LrEu[1],DtEu[1],RfEu[1]],
   'Synthatic': [LirSyn[0],GuSyn[1],LrEu[1],DtSync[1],RfSyn[1]]
   }, index=["Lin Reg","GNB","Log Reg","DT","RF"])
plt.title("Imbalanced data Test Analysis")
plt.plot(plot_data_test)


# In[269]:


fig3 = plt.figure()
fig3_sub_plot_1 = fig3.add_subplot(2,2,1)
fig3_sub_plot_2 = fig3.add_subplot(2,2,2)
plot_data.plot(title = "Imbalanced data Train Analysis" , ax = fig3_sub_plot_1)
plot_data_test.plot(title = "Imbalanced data Test Analysis" , ax = fig3_sub_plot_2)
plt.savefig('IB.png')


# In[84]:


#renaming X_test and y_test variables
test_input_europe = X_test_eu.copy()
test_output_europe = y_test_eu.copy()

test_output_europe.value_counts(normalize=True)


# In[85]:


#storing all fraud transactions
fraud_trans_eu = eu_data[eu_data['Class'] == 1]
non_fraud_trans_eu = eu_data[eu_data['Class'] == 0]

print('fraud data shape: ', fraud_trans_eu.shape)
print('non fraud data shape: ', non_fraud_trans_eu.shape)

#printing fraud data percentage
print('Fraud Data percentage: ', 100*(len(fraud_trans_eu)/len(non_fraud_trans_eu)))


# In[86]:


#random under sampling using imblearn
rusEu = RandomUnderSampler()
X_rusEu, y_rusEu = rusEu.fit_resample(X_train_eu,y_train_eu)

y_rusEu.value_counts()


# In[87]:


X_train_rusEu, X_test_rusEu, y_train_rusEu, y_test_rusEu = train_test_split(X_rusEu, y_rusEu, test_size=0.3, random_state=42, stratify=y_rusEu)
y_train_rusEu.value_counts()


# In[88]:


#renaming X_test and y_test variables
test_input_synth = y_test_syn.copy()
test_output_synth = y_test_syn.copy()
test_output_synth.value_counts(normalize=True)


# In[89]:


#storing all fraud transactions
fraud_trans_syn = syn_train_data[syn_train_data['is_fraud'] == 1]
non_fraud_trans_syn = syn_train_data[syn_train_data['is_fraud'] == 0]

print('fraud data shape: ', fraud_trans_syn.shape)
print('non fraud data shape: ', non_fraud_trans_syn.shape)

#printing fraud data percentage
print('Fraud Data percentage: ', 100*(len(fraud_trans_syn)/len(non_fraud_trans_syn)))


# In[90]:


#random under sampling using imblearn
rus_sync = RandomUnderSampler()
X_rus_sync, y_rus_sync = rus_sync.fit_resample(X_train_syn,y_train_syn)

y_rus_sync.value_counts()


# In[91]:


X_train_rusSyn, X_test_rusSyn, y_train_rusSyn, y_test_rusSyn = train_test_split(X_rus_sync, y_rus_sync, test_size=0.3, random_state=42, stratify=y_rus_sync)


# In[94]:


y_train_rusSyn.value_counts()


# In[93]:


LIREuRsu = model_implementation("LinearRegression",X_train_rusEu,X_test_rusEu,y_train_rusEu,y_test_rusEu)


# In[95]:


# add Logistic Regression - Basline
add_model_eval('LinearRegression Eu UnderSampling',LIREuRsu[2], LIREuRsu[3], LIREuRsu[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[96]:


LIRSynRsu = model_implementation("LinearRegression",X_train_rusSyn, X_test_rusSyn, y_train_rusSyn, y_test_rusSyn)


# In[97]:


# add Logistic Regression - Basline
add_model_eval('LinearRegression Syn UnderSampling',LIRSynRsu[2], LIRSynRsu[3], LIRSynRsu[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[98]:


LrEuRsu = model_implementation("LogisticRegression",X_train_rusEu,X_test_rusEu,y_train_rusEu,y_test_rusEu)


# In[99]:


# add Logistic Regression - Basline
add_model_eval('LogisticRegression Eu UnderSampling',LrEuRsu[2], LrEuRsu[3], LrEuRsu[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[100]:


LrSynRsu = model_implementation("LogisticRegression",X_train_rusSyn, X_test_rusSyn, y_train_rusSyn, y_test_rusSyn)


# In[101]:


# add Logistic Regression - Basline
add_model_eval('LogisticRegression Syn UnderSampling',LrSynRsu[2], LrSynRsu[3], LrSynRsu[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[102]:


DtEuRsu = model_implementation("DecisionTree",X_train_rusEu,X_test_rusEu,y_train_rusEu,y_test_rusEu)


# In[103]:


# add Logistic Regression - Basline
add_model_eval('DecisionTree Eu UnderSampling',DtEuRsu[2], DtEuRsu[3], DtEuRsu[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[104]:


DtSyncRsu = model_implementation("DecisionTree",X_train_rusSyn, X_test_rusSyn, y_train_rusSyn, y_test_rusSyn)


# In[105]:


# add Logistic Regression - Basline
add_model_eval('DecisionTree Syn UnderSampling',DtSyncRsu[2], DtSyncRsu[3], DtSyncRsu[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[106]:


RfEuRsu = model_implementation("RandomForestClassifier",X_train_rusEu,X_test_rusEu,y_train_rusEu,y_test_rusEu)


# In[107]:


add_model_eval('RandomForestClassifier Eu UnderSampling',RfEuRsu[2], RfEuRsu[3], RfEuRsu[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[108]:


RfSynRsu = model_implementation("RandomForestClassifier",X_train_rusSyn, X_test_rusSyn, y_train_rusSyn, y_test_rusSyn)


# In[109]:


add_model_eval('RandomForestClassifier Syn UnderSampling',RfSynRsu[2], RfSynRsu[3], RfSynRsu[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[110]:


GuEuRsu= model_implementation("GaussianNaiveBias",X_train_rusEu,X_test_rusEu,y_train_rusEu,y_test_rusEu)


# In[111]:


add_model_eval('GaussianNaiveBias Eu UnderSampling',GuEuRsu[2], GuEuRsu[3], GuEuRsu[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[112]:


GuSyncRsu= model_implementation("GaussianNaiveBias",X_train_syn,X_test_syn,y_train_syn,y_test_syn)


# In[113]:


add_model_eval('GaussianNaiveBias Syn UnderSampling',GuSyncRsu[2], GuSyncRsu[3], GuSyncRsu[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[270]:


plot_data = pd.DataFrame({
   'Europe': [LIREuRsu[0],GuEuRsu[0],LrEuRsu[0],DtEuRsu[0],RfEuRsu[0]],
   'Synthatic': [LIRSynRsu[0],GuSyncRsu[0],LrSynRsu[0],DtSyncRsu[0],RfSynRsu[0]]
   }, index=["Lin Reg","GNB","Log Reg","DT","RF"])
plt.title("Under-sampled data Train Analysis")
plt.plot(plot_data)


# In[271]:


plot_data_test = pd.DataFrame({
   'Europe': [LIREuRsu[1],GuEuRsu[1],LrEuRsu[1],DtEuRsu[1],RfEuRsu[1]],
   'Synthatic': [LIRSynRsu[1],GuSyncRsu[1],LrSynRsu[1],DtSyncRsu[1],RfSynRsu[1]]
   }, index=["Lin Reg","GNB","Log Reg","DT","RF"])
plt.title("Under-sampled data Test Analysis")
plt.plot(plot_data_test)


# In[272]:


fig3 = plt.figure()
fig3_sub_plot_1 = fig3.add_subplot(2,2,1)
fig3_sub_plot_2 = fig3.add_subplot(2,2,2)
plot_data.plot(title = "Under-sampled data Train Analysis" , ax = fig3_sub_plot_1)
plot_data_test.plot(title = "Under-sampled data Test Analysis" , ax = fig3_sub_plot_2)
plt.savefig('UnderSample.png')


# In[116]:


ros = RandomOverSampler()
X_ros_eu, y_ros_eu = ros.fit_resample(X_train_eu,y_train_eu)

y_ros_eu.value_counts()


# In[117]:


#train Test split
X_train_Euros, X_test_Euros, y_train_Euros, y_test_Euros = train_test_split(X_ros_eu,y_ros_eu, test_size=0.3, stratify=y_ros_eu, random_state=42)
y_train_Euros.value_counts()


# In[118]:


ros = RandomOverSampler()
X_ros_syn, y_ros_syn = ros.fit_resample(X_train_syn,y_train_syn)

y_ros_eu.value_counts()


# In[119]:


#train Test split
X_train_Synros, X_test_Synros, y_train_Synros, y_test_Synros = train_test_split(X_ros_eu,y_ros_eu, test_size=0.3, stratify=y_ros_eu, random_state=42)
y_train_Synros.value_counts()


# In[120]:


LIREuRos = model_implementation("LinearRegression",X_train_Euros, X_test_Euros, y_train_Euros, y_test_Euros)


# In[122]:


add_model_eval('LinearRegression Eu OverSampling',LIREuRos[2], LIREuRos[3], LIREuRos[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[123]:


LIRSynRos = model_implementation("LinearRegression",X_train_Synros, X_test_Synros, y_train_Synros, y_test_Synros )


# In[124]:


add_model_eval('LinearRegression Syn OverSampling',LIRSynRos[2], LIRSynRos[3], LIRSynRos[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[125]:


LrEuRos = model_implementation("LogisticRegression",X_train_Euros, X_test_Euros, y_train_Euros, y_test_Euros)


# In[126]:


add_model_eval('LogisticRegression Eu OverSampling',LrEuRos[2], LrEuRos[3], LrEuRos[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[127]:


LrSynRos = model_implementation("LogisticRegression",X_train_Synros, X_test_Synros, y_train_Synros, y_test_Synros )


# In[128]:


add_model_eval('LogisticRegression Syn OverSampling',LrSynRos[2], LrSynRos[3], LrSynRos[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[277]:


DtEuRos = model_implementation("DecisionTree",X_train_Euros, X_test_Euros, y_train_Euros, y_test_Euros)


# In[130]:


add_model_eval('DecisionTree Eu OverSampling',DtEuRos[2], DtEuRos[3], DtEuRos[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[278]:


DtSynRos = model_implementation("DecisionTree",X_train_Synros, X_test_Synros, y_train_Synros, y_test_Synros )


# In[132]:


add_model_eval('DecisionTree Syn OverSampling',DtSynRos[2], DtSynRos[3], DtSynRos[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[133]:


RfEuRos = model_implementation("RandomForestClassifier",X_train_Euros, X_test_Euros, y_train_Euros, y_test_Euros)


# In[134]:


add_model_eval('RandomForestClassifier Eu OverSampling',RfEuRos[2], RfEuRos[3], RfEuRos[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[135]:


RfSynRos = model_implementation("RandomForestClassifier",X_train_Synros, X_test_Synros, y_train_Synros, y_test_Synros )


# In[136]:


add_model_eval('RandomForestClassifier Syn OverSampling',RfSynRos[2], RfSynRos[3], RfSynRos[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[137]:


GuEuRos= model_implementation("GaussianNaiveBias",X_train_Euros, X_test_Euros, y_train_Euros, y_test_Euros)


# In[138]:


add_model_eval('GaussianNaiveBias Eu OverSampling',GuEuRos[2], GuEuRos[3], GuEuRos[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[139]:


GuSyncRos= model_implementation("GaussianNaiveBias",X_train_Synros, X_test_Synros, y_train_Synros, y_test_Synros )


# In[140]:


add_model_eval('GaussianNaiveBias Syn OverSampling',GuSyncRos[2], GuSyncRos[3], GuSyncRos[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[279]:


plot_data = pd.DataFrame({
   'Europe': [LIREuRos[0],GuEuRos[0],LrEuRos[0],DtEuRos[0],RfEuRos[0]],
   'Synthatic': [LIRSynRos[0],GuSyncRos[0],LrSynRos[0],DtSynRos[0],RfSynRos[0]]
   }, index=["Lin Reg","GNB","Log Reg","DT","RF"])
plt.title("Over-sample training dataset")
plt.plot(plot_data)


# In[280]:


plot_test_data = pd.DataFrame({
   'Europe': [LIREuRos[1],GuEuRos[1],LrEuRos[1],DtEuRos[1],RfEuRos[1]],
   'Synthatic': [LIRSynRos[1],GuSyncRos[1],LrSynRos[1],DtSynRos[1],RfSynRos[1]]
   }, index=["Lin Reg","GNB","Log Reg","DT","RF"])
plt.title("Over-sample test dataset")
plt.plot(plot_test_data)


# In[281]:


fig3 = plt.figure()
fig3_sub_plot_1 = fig3.add_subplot(2,2,1)
fig3_sub_plot_2 = fig3.add_subplot(2,2,2)
plot_data.plot(title = "Over-sampled data Train Analysis" , ax = fig3_sub_plot_1)
plot_test_data.plot(title = "Over-sampled data Test Analysis" , ax = fig3_sub_plot_2)
plt.savefig('OverSample.png')


# In[144]:


#balancing using SMOTE method
smote = SMOTE(sampling_strategy='minority')
X_sm_eu, y_sm_eu = smote.fit_resample(X_train_eu.astype('float'), y_train_eu)

y_sm_eu.value_counts()


# In[145]:


#train test split
X_train_eusm, X_test_eusm, y_train_eusm, y_test_eusm = train_test_split(X_sm_eu, y_sm_eu, test_size=0.3, random_state=42, stratify=y_sm_eu)
y_train_eusm.value_counts()


# In[146]:


#balancing using SMOTE method
smote = SMOTE(sampling_strategy='minority')
X_sm_syn, y_sm_syn = smote.fit_resample(X_train_syn.astype('float'), y_train_syn)

y_sm_syn.value_counts()


# In[147]:


#train test split
X_train_synsm, X_test_synsm, y_train_synsm, y_test_synsm = train_test_split(X_sm_syn, y_sm_syn, test_size=0.3, random_state=42, stratify=y_sm_syn)
y_train_synsm.value_counts()


# In[148]:


LIREuSm = model_implementation("LinearRegression",X_train_eusm, X_test_eusm, y_train_eusm, y_test_eusm)


# In[149]:


add_model_eval('LinearRegression Eu SMOTE',LIREuSm[2], LIREuSm[3], LIREuSm[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[150]:


LIRSynSm = model_implementation("LinearRegression",X_train_synsm, X_test_synsm, y_train_synsm, y_test_synsm)


# In[151]:


add_model_eval('LinearRegression Syn SMOTE',LIRSynSm[2], LIRSynSm[3], LIRSynSm[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[152]:


LrEuSm = model_implementation("LogisticRegression",X_train_eusm, X_test_eusm, y_train_eusm, y_test_eusm)


# In[153]:


add_model_eval('LogisticRegression Eu SMOTE',LrEuSm[2], LrEuSm[3], LrEuSm[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[154]:


LrSynSm = model_implementation("LogisticRegression",X_train_synsm, X_test_synsm, y_train_synsm, y_test_synsm)


# In[155]:


add_model_eval('LogisticRegression Syn SMOTE',LrSynSm[2], LrSynSm[3], LrSynSm[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[252]:


DtEuSm = model_implementation("DecisionTree",X_train_eusm, X_test_eusm, y_train_eusm, y_test_eusm)


# In[157]:


add_model_eval('DecisionTree Eu SMOTE',DtEuSm[2], DtEuSm[3], DtEuSm[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[251]:


DtSynRos = model_implementation("DecisionTree",X_train_synsm, X_test_synsm, y_train_synsm, y_test_synsm)


# In[159]:


add_model_eval('DecisionTree Syn SMOTE',DtSynRos[2], DtSynRos[3], DtSynRos[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[160]:


RfEuSm = model_implementation("RandomForestClassifier",X_train_eusm, X_test_eusm, y_train_eusm, y_test_eusm)


# In[161]:


add_model_eval('RandomForestClassifier Eu SMOTE',RfEuSm[2], RfEuSm[3], RfEuSm[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[162]:


RfSynSm = model_implementation("RandomForestClassifier",X_train_synsm, X_test_synsm, y_train_synsm, y_test_synsm)


# In[163]:


add_model_eval('RandomForestClassifier Syn SMOTE',RfSynSm[2], RfSynSm[3], RfSynSm[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[164]:


GuEuSm= model_implementation("GaussianNaiveBias",X_train_eusm, X_test_eusm, y_train_eusm, y_test_eusm)


# In[165]:


add_model_eval('GaussianNaiveBias Eu SMOTE',GuEuSm[2], GuEuSm[3], GuEuSm[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[166]:


GuSynSm= model_implementation("GaussianNaiveBias",X_train_synsm, X_test_synsm, y_train_synsm, y_test_synsm)


# In[167]:


add_model_eval('GaussianNaiveBias Syn SMOTE',GuSynSm[2], GuSynSm[3], GuSynSm[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[274]:


plot_data = pd.DataFrame({
   'Europe': [LIREuSm[0],GuEuSm[0],LrEuSm[0],DtEuSm[0],RfEuSm[0]],
   'Synthatic': [LIRSynSm[0],GuSynSm[0],LrSynSm[0],DtSynRos[0],RfSynSm[0]]
   }, index=["Lin Reg","GNB","Log Reg","DT","RF"])
plt.title("SMOTE Train Dataset")
plt.plot(plot_data)


# In[275]:


plot_Test_data = pd.DataFrame({
   'Europe': [LIREuSm[1],GuEuSm[1],LrEuSm[1],DtEuSm[1],RfEuSm[1]],
   'Synthatic': [LIRSynSm[1],GuSynSm[1],LrSynSm[1],DtSynRos[1],RfSynSm[1]]
   }, index=["Lin Reg","GNB","Log Reg","DT","RF"])
plt.title("SMOTE Test Dataset")
plt.plot(plot_Test_data)


# In[276]:


fig3 = plt.figure()
fig3_sub_plot_1 = fig3.add_subplot(2,2,1)
fig3_sub_plot_2 = fig3.add_subplot(2,2,2)
plot_data.plot(title = "SMOTE Train Dataset" , ax = fig3_sub_plot_1)
plot_Test_data.plot(title = "SMOTE Test Dataset" , ax = fig3_sub_plot_2)
fig3.savefig('smote.png')


# In[170]:


def deepLearingModel(model,X_train,X_test,y_train,y_test):
    if model == "NeuralNetwork":
        model = "Neural Network"
        modelImpl = classifierCreator()
        modelImpl.compile(loss='binary_crossentropy', optimizer='sgd')
        weights = {0:1, 1:1.5}
        
    historyEu = modelImpl.fit(X_train, y_train, class_weight=weights, epochs=15, verbose=0)
    pred_train = (modelImpl.predict(X_train)> 0.5).astype("int32")
    pred_test = (modelImpl.predict(X_test)> 0.5).astype("int32")

    accuracy_train = accuracy_score(pred_train,y_train)
    accuracy_test = accuracy_score(pred_test,y_test)
    confusionMatrix = confusion_matrix(y_test,pred_test)
    classificationReport = classification_report(y_test,pred_test)
    mae = mean_absolute_error(y_train,pred_train)
    roc_train = roc(y_train, pred_train)
    roc_test = roc(y_test, pred_test)
    f1_score = metrics.f1_score(y_test,pred_test)
    recall = metrics.recall_score(y_test, pred_test)
    print("Model Implemented: ", model)
    print("Accuracy on Training Set: ", accuracy_train)
    print('Accuracy on Validation Set: ', accuracy_test)
    print('ROC AUC Score Train: ',roc(y_train, pred_train))
    print('ROC AUC Score Test: ',roc(y_test, pred_test))
    print(f'(train) MCC: {metrics.matthews_corrcoef(y_train, pred_train)}')
    print(f'(train) F1: {metrics.f1_score(y_train,pred_train)}')
    print(f'(train) Recall: {metrics.recall_score(y_train, pred_train)}')
    print('Mean absolute error: ',mae)
    print('Confusion Matrix\n', confusionMatrix)
    print('Classification Report\n', classificationReport)
    fig, (ax1) = plt.subplots(1, figsize = (10, 4))
    sns.heatmap(confusionMatrix, annot=True,ax = ax1, fmt = 'g')
    ax1.set_xlabel('Predicted labels');ax1.set_ylabel('True labels') 
    ax1.set_title('Confusion Matrix before Fine tuning')
    ax1.xaxis.set_ticklabels(['Non Fraudulent', 'Fradulent']); ax1.yaxis.set_ticklabels(['Non Fraudulent', 'Fradulent'])
    

    return [accuracy_train,accuracy_test,roc_test,recall,f1_score]


# In[172]:


#create classifier Neural Network Classifier
def classifierCreator():
    clf = keras.models.Sequential()
    clf.add(keras.layers.Dense(10, activation='relu'))
    clf.add(keras.layers.Dense(5, activation='relu'))
    clf.add(keras.layers.Dense(1, activation='sigmoid'))
    return clf


# In[173]:


deepEuNorm = deepLearingModel("NeuralNetwork",X_train_eu,X_test_eu,y_train_eu,y_test_eu)


# In[174]:


add_model_eval('NeuralNetwork Eu imbalance',deepEuNorm[2], deepEuNorm[3], deepEuNorm[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[175]:


deepSynNorm = deepLearingModel("NeuralNetwork",X_train_syn,X_test_syn,y_train_syn,y_test_syn)


# In[176]:


add_model_eval('NeuralNetwork Syn imbalance',deepSynNorm[2], deepSynNorm[3], deepSynNorm[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[177]:


deepEuRsu =deepLearingModel("NeuralNetwork",X_train_rusEu,X_test_rusEu,y_train_rusEu,y_test_rusEu)


# In[178]:


add_model_eval('NeuralNetwork Eu undersample',deepEuRsu[2], deepEuRsu[3], deepEuRsu[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[179]:


deepSynRsu =deepLearingModel("NeuralNetwork",X_train_syn,X_test_syn,y_train_syn,y_test_syn)


# In[180]:


add_model_eval('NeuralNetwork Syn undersample',deepSynRsu[2], deepSynRsu[3], deepSynRsu[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[181]:


deepEuRos =deepLearingModel("NeuralNetwork",X_train_Euros, X_test_Euros, y_train_Euros, y_test_Euros)


# In[182]:


add_model_eval('NeuralNetwork Eu Oversample',deepEuRos[2], deepEuRos[3], deepEuRos[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[183]:


deepSynRos = deepLearingModel("NeuralNetwork",X_train_Synros, X_test_Synros, y_train_Synros, y_test_Synros )


# In[184]:


add_model_eval('NeuralNetwork Syn Oversample',deepSynRos[2], deepSynRos[3], deepSynRos[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[185]:


deepEuSm =deepLearingModel("NeuralNetwork",X_train_eusm, X_test_eusm, y_train_eusm, y_test_eusm)


# In[186]:


add_model_eval('NeuralNetwork Eu SMOTE',deepEuSm[2], deepEuSm[3], deepEuSm[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[187]:


deepSynSm = deepLearingModel("NeuralNetwork",X_train_synsm, X_test_synsm, y_train_synsm, y_test_synsm)


# In[188]:


add_model_eval('NeuralNetwork Syn SMOTE',deepSynSm[2], deepSynSm[3], deepSynSm[4])
df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[282]:


plot_data = pd.DataFrame({
   'Europe': [deepEuNorm[0],deepEuRsu[0],deepEuRos[0],deepEuSm[0]],
   'Synthatic': [deepSynNorm[0],deepSynRsu[0],deepSynRos[0],deepSynSm[0]]
   }, index=["No Sample","Under Sample","Over Sample","SMOTE"])
plt.title("Neural Network")
plt.plot(plot_data)


# In[283]:


plot_test_data = pd.DataFrame({
   'Europe': [deepEuNorm[1],deepEuRsu[1],deepEuRos[1],deepEuSm[1]],
   'Synthatic': [deepSynNorm[1],deepSynRsu[1],deepSynRos[1],deepSynSm[1]]
   }, index=["No Sample","Under Sample","Over Sample","SMOTE"] )
# plt.title("Neural Network Test")
# lines = df.plot.line()
# plt.plot(plot_test_data)
# plot_test_data.plot.title("Neural Network Test")
# plot_test_data.plot.line()
ax = plt.gca()
plot_test_data.plot(ax = ax)


# In[284]:


fig3 = plt.figure()
fig3_sub_plot_1 = fig3.add_subplot(2,2,1)
fig3_sub_plot_2 = fig3.add_subplot(2,2,2)
plot_data.plot(title = "Neural Network" , ax = fig3_sub_plot_1)
plot_test_data.plot(title = "Neural Network Test" , ax = fig3_sub_plot_2)
plt.savefig('NN.png')


# In[265]:


df_metrics_table = view_models_eval(df_metrics_table)
display(df_metrics_table)


# In[ ]:




