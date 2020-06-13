#!/usr/bin/env python
# coding: utf-8

# In this analysis, I have done the exploratory analysis on Abalone Dataset. Post that I tested the scores for different 
# Regression Algorithms.
# Dataset can be found on https://archive.ics.uci.edu/ml/datasets.php

# Data Set Information:

# Predicting the age of abalone from physical measurements. The age of abalone is determined by cutting the shell through 
# the cone, staining it, and counting the number of rings through a microscope -- a boring and time-consuming task. Other
# measurements, which are easier to obtain, are used to predict the age. Further information, such as weather patterns and 
# location (hence food availability) may be required to solve the problem.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

#Sex / nominal / -- / M, F, and I (infant)
#Length / continuous / mm / Longest shell measurement
#Diameter / continuous / mm / perpendicular to length
#Height / continuous / mm / with meat in shell
#Whole weight / continuous / grams / whole abalone
#Shucked weight / continuous / grams / weight of meat
#Viscera weight / continuous / grams / gut weight (after bleeding)
#Shell weight / continuous / grams / after being dried
#Rings / integer / -- / +1.5 gives the age in years

# Load the dataset, remove headers and name the columns
df=pd.read_csv("D:/abalone.data", header=None,
              names=['Sex','Length','Diameter','Height','Whole_weight','Shucked_weight','Viscera weight','Shell_weight',
                    'Ring'])
df.head(2)

# For age in years
df['Age'] = df['Ring']+1.5
df.head(2)

# Check the info about dataset
df.info()
# We have Sex as an object, rest all as float

# Check if dataset has any null values
df.isnull().sum()

df.corr()
# Seems all parameters are well correlated with the output field so we will consider all the parameters in the training

df.describe()

# Let us Analyse the Output field
sb.distplot(df['Ring'])

sb.boxplot(df['Ring'])

# from the above plots, we can see that the normal spread is spread in the range of 2Standard Deviation approximately
# Max: 9.933684+2*3.224169 = 16.382022
# Min: 9.933684-2*3.224169 = 3.485346....abovve box plot shows this fact

# We plot the parameters related to the dimension: Length, Diameter and Height

sb.pairplot(df[['Length','Diameter','Height']])
# Here we see that Height is showing some outliers, so let us check for outliers

# Try to see the outliers in Height
sb.boxplot(df['Height']) #Shows outliers way beyond the major spread range

# To better visualize the distribution, we cut the x limits to the major spread of height
s = sb.distplot(df['Height'])
s.set_xlim(0.0,0.26) #setting the range (started with 3, setteled with 0.26 as it shows the maximum distribution)

# Similarly, Analysing the Weight factors: Whole_weight, Shucked_weight, Viscera weight, Shell_weight
sb.pairplot(df[['Whole_weight', 'Shucked_weight', 'Viscera weight', 'Shell_weight']])
# We see that distribution area of weight parameters is a bit larger than the dimension parameters

# Multivariate Analysis: Here we visualize how all parameters are correlated to the output parameter. POst this, we will
# explore the most correlated factors
pt = df.corr()
plt.figure(figsize=(8,8))
sb.heatmap(pt, annot=True)

# In the above plot we see that Shell_weight and Diameter are the most strongly correlated to the Ring, so we plot a 
# joint plot to see the density and the scatter plot both

sb.jointplot(data=df, x='Ring', y='Shell_weight', kind='reg')
sb.jointplot(data=df, x='Ring', y='Diameter', kind='reg')

# From the above plot we observed that the shell weight and the diameter remains concentrated below certain level till the
# the ring size of 5 and starts increasing with ring value 5. Further more, with larger values of ring, these two parameters
# start dispersing

# Let us check how the correlation varies with the number of rings
# Ring size 0 to 10 shows a different pattern as compared to above ring size > 10
# Let us visualize the corelation in these two segments

# Segment-1: Ring size<10
df1 = df[df['Ring']<10]
pt = df1.corr()
plt.figure(figsize=(8,8))
sb.heatmap(pt, annot=True)

# Noe let us examine the dimension parameters w.r.t rings as they seems more stromgly correlated to rings
sb.jointplot(df1['Ring'], df1['Length'], kind='reg')
sb.jointplot(df1['Ring'], df1['Diameter'], kind='reg')
sb.jointplot(df1['Ring'], df1['Height'], kind='reg')

# Above plots shows that all 3 Dimension parameters starts increasing after the ring size 3

# Segment-2: Ring size>=10
df2 = df[df['Ring']>=10]
pt=df2.corr()
plt.figure(figsize=(8,8))
sb.heatmap(pt, annot=True)

# We clearly see that the correlation of all parameters greatly degrades with more ring size

# Check the weight, height and diameter (just taking these 3 parameters as they wer most correlated to ring size)
sb.jointplot(df2['Ring'], df2['Shell_weight'], kind='reg')
sb.jointplot(df2['Ring'], df2['Height'], kind='reg')
sb.jointplot(df2['Ring'], df2['Diameter'], kind='reg')

# From the above plots, it is clear that the Abalons increase in dimension and weight till the ring size of 10 but after the 
# ring size of 10 to 11, it shows spreading or dispersing pattern so we can say that after ring size of 10 to 11, the development 
# in size and weight depends on environmental factors, which has no significant pattern

# Now let us examine the SEX w.r.t Ring size

# Sex: Thres categories are found : Male, Female and Infant(Which are not adult)
df['Sex'].unique()

df['Sex'].value_counts()

# Visualizing the counts of each category of Sex
sb.countplot(df['Sex'])

#PLotting the distributin of SEX w.r.t different parameters (I took the major parameters correlated to rings)
sb.boxplot(data=df,x='Ring', y='Sex')

sb.boxplot(data=df,x='Shell_weight', y='Sex')

sb.boxplot(data=df,x='Diameter', y='Sex')

sb.boxplot(data=df,x='Height', y='Sex')

# From al the above plots, we get to know that the median of different parameters for INFANTS are lower as compared to the
# MAle/Female. Distribution of all three categories are concentrated around the mean.

# Let us check te correlation of SEX w.r.t. most storngly correlated parameters "Diameter, Height and Shell_weight"
# Why these parameters?: As these 3 parameters shown a Strong Correlation with lower values of Rings and Infants has lower
# Values as compared to Male/Female. We will plot the Linear Regression plot woth variations in SEX with the help of "HUE"

sb.lmplot(data=df, x='Ring', y='Height', hue='Sex')
sb.lmplot(data=df, x='Ring', y='Diameter', hue='Sex')
sb.lmplot(data=df, x='Ring', y='Shell_weight', hue='Sex')

# Observing all the above lmplots, we see that the Infant line is more inclined as compared to MAale/Female. 
# Plot shows strong correlation of parameters with ring
# Building Regression Models: Applying different classification algorithms
# Preparing dataset

df['Sex_E'] = df['Sex'].replace({'M':1, 'F':2, 'I':3})
df.head(2)

df1 = df.drop(['Sex'], axis=1)
df1.head(2)

dfi = df1.iloc[:, df1.columns != 'Ring']
dfo= df1['Ring']

# Implementing Algorithms
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

lr = LinearRegression()
svr_lin = SVR(kernel='linear')
svr_rbf=SVR(kernel='rbf')

lr=LinearRegression()
lr.fit(dfi,dfo)
lr.score(dfi,dfo)
# 1.0
svr_lin = SVR(kernel='linear')
svr_lin.fit(dfi,dfo)
svr_lin.score(dfi,dfo)
# 0.9997250282825754
svr_rbf=SVR(kernel='rbf')
svr_rbf.fit(dfi,dfo)
svr_rbf.score(dfi,dfo)
# 0.9905376480575866
