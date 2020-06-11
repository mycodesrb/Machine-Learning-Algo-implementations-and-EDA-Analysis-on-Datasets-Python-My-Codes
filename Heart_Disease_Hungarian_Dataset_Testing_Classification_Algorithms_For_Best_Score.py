#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In this analysis, I have cleaned dataset, prepared it to tran algorithms and then I tested the scores for different 
# Classification Algorithms.

# Dataset can be found on https://archive.ics.uci.edu/ml/datasets.php

# Data Set Information:

# This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular,
# the Cleveland database is the only one that has been used by ML researchers to this date. We have two datasets, Hungarian
# and Switzerland. Approach for both are the same. I have done the coding on Hungarian dataset only.
# The "goal" field refers to the presence of heart disease in the patient.   


# In[2]:


# Attribute Information:
#(age) 
#(sex) (1 = male; 0 = female)
#(cp) : chest pain type
#Value 1: typical angina
#Value 2: atypical angina
#Value 3: non-anginal pain
#Value 4: asymptomatic

#(trestbps): resting blood pressure (in mm Hg on admission to the hospital)
#(chol) : serum cholestoral in mg/dl
#(fbs): (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
#(restecg):  resting electrocardiographic results
#Value 0: normal
#Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
#Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria

#(thalach): maximum heart rate achieved
#(exang): exercise induced angina (1 = yes; 0 = no)
#(oldpeak): ST depression induced by exercise relative to rest
#(slope): the slope of the peak exercise ST segment
#Value 1: upsloping
#Value 2: flat
#Value 3: downsloping

#(ca): number of major vessels (0-3) colored by flourosopy
#(thal): 3 = normal; 6 = fixed defect; 7 = reversable defect
#(num) (the predicted attribute) diagnosis of heart disease (angiographic disease status)
#Value 0: < 50% diameter narrowing: Absense
#Value 1: > 50% diameter narrowing: Presense


# In[3]:


import pandas as pd


# In[4]:


# Load the dataset, remove headers, provide names to the attributes
df =pd.read_csv("d:/uhungarian.data", header = None, names=['age','sex','cp','trestbps','chol','fbs','restecg','thalach',
                                                             'exang','oldpeak','slope','ca','thal','num'])
df.head(2)


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.isnull().sum() # no specific null values but we need to check other values/symbols which is not of out interest


# In[8]:


# We will check each attribute, clean it if required and we will encode it to get it in desired input form for an algorithm
df['trestbps'].value_counts() # found just one '?' symbol, as we will replace it with mean value  


# In[9]:


df['trestbps'] = df['trestbps'].replace("?",'NaN')
df['trestbps'] = df['trestbps'].astype('float')
mean_trestbps = round(df['trestbps'].mean())
df['En_trestbps'] = df['trestbps'].fillna(mean_trestbps) #to persists the change, we have to save it in another column
print(mean_trestbps)
df['En_trestbps'].value_counts()


# In[10]:


df['chol'].value_counts() #As we have majority values as '?', we can replace it with the mode value


# In[11]:


from sklearn.preprocessing import Imputer # used to replace NaN values with desired values
df['chol'] = df['chol'].replace("?",'NaN') # Make "?" as NaN
imp=Imputer(missing_values="NaN", strategy="most_frequent" ) 
df["En_chol"]=imp.fit_transform(df[["chol"]]).ravel()
df["En_chol"].value_counts()


# In[12]:


df['fbs'].value_counts() # '?' found, less in counts, so we will replace it with mean value


# In[13]:


df['fbs'] = df['fbs'].replace("?", 'NaN')
df['fbs'] = df['fbs'].astype('float')
mean_fbs = round(df['fbs'].mean())
df['En_fbs'] = df['fbs'].fillna(mean_fbs)
df['En_fbs'].value_counts()


# In[14]:


df['restecg'].value_counts() # just one '?', replace with 0 as it has the max frequency


# In[15]:


df['restecg'] = df['restecg'].replace("?",'NaN')
df['restecg'] = df['restecg'].astype("float")
mean_rest = round(df['restecg'].mean())
df['En_restecg'] = df['restecg'].fillna(mean_rest)
df['En_restecg'].value_counts()


# In[16]:


df['thalach'].value_counts() # one '?', replace with mean


# In[17]:


df['thalach'] = df['thalach'].replace("?",'NaN')
df['thalach'] = df['thalach'].astype('float')
mean_thalach= round(df['thalach'].mean())

df['En_thalach'] = df['thalach'].fillna(mean_thalach)
df['En_thalach'].value_counts()


# In[18]:


df['exang'].value_counts() # one "?", replace with mean


# In[19]:


df['exang'] = df['exang'].replace("?",'NaN')
df['exang'] = df['exang'].astype('float')
mean_exang= round(df['exang'].mean())

df['En_exang'] = df['exang'].fillna(mean_exang)
df['En_exang'].value_counts()


# In[20]:


df['oldpeak'].value_counts() # perfect 


# In[21]:


df['slope'].value_counts() # MAximum '?', use Imputer


# In[22]:


df['slope'] = df['slope'].replace("?",'NaN') # Make "?" as NaN
imp=Imputer(missing_values="NaN", strategy="mean" ) 
df["En_slope"]=imp.fit_transform(df[["slope"]]).ravel()
df["En_slope"].value_counts()


# In[23]:


df['ca'].value_counts() # most of all are '?', only 3 are different, we can drop this column also but we will prefer to keep it


# In[24]:


df['ca'] = df['ca'].replace("?",'NaN')
df['ca'] = df['ca'].astype('float')
mean_ca= round(df['ca'].mean())

df['En_ca'] = df['ca'].fillna(mean_ca)
df['En_ca'].value_counts()


# In[25]:


df['thal'].value_counts() #replac '?' with mean


# In[26]:


df['thal'] = df['thal'].replace("?",'NaN')
df['thal'] = df['thal'].astype('float')
mean_thal= round(df['thal'].mean())

df['En_thal'] = df['thal'].fillna(mean_thal)
df['En_thal'].value_counts()


# In[27]:


# Prepare final dataframe by removing the unwanted attributes keeping all encoded attributes
df1=df.drop(['trestbps','chol','fbs','restecg','thalach','exang','slope','ca','thal'], axis=1)
# Final encoded dataset 
df1.head(2)


# In[28]:


# prepare input and output
dfi = df1.iloc[:,df1.columns!='num']
dfo = df1['num']


# In[29]:


#split in train and test
from sklearn.model_selection import train_test_split


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(dfi,dfo, test_size=0.30)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# In[31]:


# Fit and get scores
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

algos = {'Logistic Regression':LogisticRegression(), 'Decision Tree':DecisionTreeClassifier(), 'Random Forest':RandomForestClassifier(),
            'Naive Bayes':GaussianNB(),'KNN':KNeighborsClassifier(), 'SVC_Rbf':SVC(kernel='rbf') }


# In[32]:


lsnames, lsscores=[],[]
for i, j in algos.items():
    j.fit(X_train, y_train)
    
    lsnames.append(i)
    lsscores.append(j.score(X_train, y_train))
        
print('_____________Algorithm Scores______________')
for a in range(len(lsnames)):
    print(lsnames[a]," : ", lsscores[a])
    print('===========================================')
    


# In[33]:


# Comparing the scores, the best suited algorithm can be decided. Further more, confusion matrix and classification report
# can also be compared, if required.

