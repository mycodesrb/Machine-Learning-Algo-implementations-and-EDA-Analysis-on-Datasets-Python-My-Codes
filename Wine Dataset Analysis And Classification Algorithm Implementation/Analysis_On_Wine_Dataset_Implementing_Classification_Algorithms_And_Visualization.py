#!/usr/bin/env python
# coding: utf-8

# Here I will try to do the analysis on popular Wine dataset to understand the dataset indepth.
# Data Set Information:
#These data are the results of a chemical analysis of wines grown in the same region in Italy but derived from three different
# cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines.
# Class: The goal column
# Input Columns :
#1) Alcohol
#2) Malic acid
#3) Ash
#4) Alcalinity of ash
#5) Magnesium
#6) Total phenols
#7) Flavanoids
#8) Nonflavanoid phenols
#9) Proanthocyanins
#10)Color intensity
#11)Hue
#12)OD280/OD315 of diluted wines
#13)Proline

# In a classification context, this is a well posed problem with "well behaved" class structures. A good data set for first 
# testing of a new classifier.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

#load Data
df = pd.read_csv("d:/wine.data", header=None, names=['Class','Alcohol','M_acid','Ash','Alcalinity','Magnesium','Tot_phenols',
                             'Flavanoids','Nonflav','Proanthocyanins','Colorint','Hue','OD','Proline'])
df.head()

df.shape

f = open("d:/wine.names",'r')
for line in f.readlines():
    print(line)

df.info()

df.isna().sum()

df.head(2)

# Output Column: Let us consider as Score 1,2 & 3
print(df['Class'].value_counts())
sb.countplot(df['Class'])
plt.show()
# Most of the wines are present in the Category of Score 2 

df['Proline'].value_counts()

# Other Columns
df.describe()

pt = df.corr()
fig, ax1 = plt.subplots(figsize=(10,8))
sb.heatmap(pt, ax=ax1, annot=True)
plt.show()

# I the above heatmap, we see that the maximum positively correlated factors for the wine class are Alcalinity followed by 
# Nonflav but maximum inversly correlated are Flavnoids, OD and Tot_Phenols

# Checking correlation
df[['Class','Alcalinity','Nonflav']].corr()

# Checking pairplot of Highest correlated factors
sb.pairplot(df[['Class','Alcalinity','Nonflav']])

# Above correlation and plots shows that with increase in 'Alcalinity' and 'Nonflav' will affect proportionally to Class 

# For example, lets crosscheck the density of Acalinity w.r.t Class Score
sb.factorplot(data=df, x='Class', y='Alcalinity')
plt.show()
# The factorplot also shows the positive correlation of Alcalinity with Class Score

# Checking highest negative correlation values
df[['Class','Flavanoids','OD','Tot_phenols']].corr()

# Checking pairplot of Highest negative correlated factors
sb.pairplot(df[['Class','Flavanoids','OD','Tot_phenols']])

# Above correlation and plots shows that with increase in 'Flavanoids','OD' and 'Tot_phenols' will affect the Class score in 
# inversely proportional way

# For example, let's crosscheck by factorplot for 'Class' vs Flavanoids'
sb.factorplot(data=df, x='Class',y='Flavanoids')
plt.show()
# Plot show the inverse proportionality with Class Score

# Let's implement some classification Algorithms and check the corresponding scores
# Further more, train and test split can also be done as I have done in many other algorithms in ML repository, here I am doing
# the normal way

# Make Inputs and Output
dfi = df.iloc[:,1:]
dfo=df['Class']

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

lr = LogisticRegression()
lr.fit(dfi,dfo)
print("Score for Logistic Regression: ",lr.score(dfi,dfo))

kn=KNeighborsClassifier()
kn.fit(dfi,dfo)
print("Score for KNN: ",kn.score(dfi,dfo))

nb = GaussianNB()
nb.fit(dfi,dfo)
print("Score for Naive bayes: ",nb.score(dfi,dfo))

dtc= DecisionTreeClassifier()
dtc.fit(dfi,dfo)
print("Score For Decision Tree Classification: ",dtc.score(dfi,dfo))

svc_lin = SVC(kernel='linear')
svc_rbf = SVC(kernel='rbf')

svc_lin.fit(dfi,dfo)
print("Score for SVC_Linear: ", svc_lin.score(dfi,dfo))

svc_rbf.fit(dfi,dfo)
print("Score for SVC_RBF: ",svc_rbf.score(dfi,dfo))

# Also, for more deep checking the scores precision and recall wise, confusion matrix and classification report can be studied.
# Depending upon need the best scored algorithm can be finalized for the prediction.

# Let us build a confusion matrix and classification report for any one algoritghm. We can do the same process for all the
# classification algorithms: Let us do it on KNN
# Import the required:
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#Predict with KNN
df['Class_knn'] = kn.predict(dfi)
df.head(2)

# Create confusion Matrix and Classification Report
print(confusion_matrix(df['Class'],df['Class_knn']))
print(classification_report(df['Class'],df['Class_knn']))
# Many conclusions can be derived and considered for the consideration of score like Clasification shows that row wise accuracy
# is more as compared to column wise accuracy etc.

# Let us visualize different classes on any two parameters, say Alcalinity and Hue
for a in df['Class'].unique():
    dfc=df[df['Class']==a]
    plt.scatter(dfc['Alcalinity'],dfc['Hue'],label="Class "+str(a))
    plt.scatter(dfc['Alcalinity'],dfc['Hue'],label="Class "+str(a))
plt.legend()
plt.show()

# we can also cluster out the dataset on the basis of some values by the application of K Means algorithm if required.
