#!/usr/bin/env python
# coding: utf-8

# Here i will analyse the Mall Customer Dataset on its variables and then implement the KNN algorithm. Also, i will include
# the visualizations.
# Dataset: It has 5 Columns Viz. CustomerID, Gender, Age, Annual Income (k$) and Spending Score (1-100). Spending Score is a 
# Score which is provided on the basis of Customer's Behaviour and Spendng Nature.

# import required libs and load the dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv("d:/Mall_Customers_KMeans.csv")
df.head()

df.shape

df.info() # all are numeric except Age

# Check for null: 
df.isna().sum() # No null values

plt.hist(df['CustomerID']) # all unique

print(df['Gender'].value_counts())
sb.countplot(df['Gender']) # more female customers as compared to males

df['Age'].describe() # MAximum Age 70, minimum age 70, seems ages >=49 are in 75% of all popuplation. Spread is at 75% so a 
# little right skewed is expected

sb.boxplot(df['Age'])
# Skewed to right slightly. Maximum data is present in the 3rd Quartile. Mean < Median. No outliers were found
# After age of 50, very little customer may exist

sb.distplot(df['Age'])
# As comapred to other age groups, Very little customers came for shopping from the age group >50
# Maxium crowed came for Shopping belongs to age group between 35 to <40
# Least shopping done by younsters and old people, middle aged people seems to shop more here

df['Annual Income (k$)'].describe() # Minimum earning, 15k$ and MAx is 137k$, maximum people (75%) earns better as compared to 
# minimum salary

plt.hist(df['Annual Income (k$)']) # Top earnings are between 45K$ to 85K$. Very less people earn more than 85K$

sb.boxplot(df['Annual Income (k$)']) #Mean and median are near to each other. Few Outliers are present
# Very slight skewed to right

sb.distplot(df['Annual Income (k$)'])

#print(len(df['Spending Score (1-100)'].unique()) # 84 unique scores)
df['Spending Score (1-100)'].describe()

plt.hist(df['Spending Score (1-100)'])
# Frequency of customers who got Highest Scores (between 40 to 60) are high

sb.pairplot(df[['Gender','Age','Annual Income (k$)','Spending Score (1-100)']])

sb.heatmap(df.corr(), annot=True) # no values are tightly correlated 

plt.scatter(df['Spending Score (1-100)'],df['Age'])
plt.xlabel("Score")
plt.ylabel("Age")
plt.title("Age vs Score Scatterplot")
plt.show()
# Here we see that people in the age group 35 to 60 has maximum scores followed by poeple up to age 20 but post 60 years,
# the scores of people is quite low as compared

sb.boxenplot(df['Gender'], df['Spending Score (1-100)'])
plt.title("Gender Vs Score")
plt.show()
# It shows that most of the males have the spending scores between 25 to 70 whereas Females have between 35 to 70 which shows
# that Females are having more shopping score, hence more shopping

#sb.barplot(data=df, y='Annual Income (k$)',x='Gender',hue='Gender')
sb.violinplot(df['Gender'], df['Annual Income (k$)'])
plt.title("Gender vs Annual Income")
plt.show()
# Plot shows that more Males are getting a bigger Annual Income as compared to Females. 
# As far as low annual incomes are concerned, Male and Female seems both same 

from sklearn.cluster import KMeans

# Kmeans implementation for clustering
df.head(2)
df['Gender1'] = df['Gender'].replace({'Male':0, 'Female':1}) # Encode Gender in numeric form
df.head(2)
df1 = df.drop(['CustomerID','Gender'], axis=1) # remove unwanted columns
df1.head(2)

dfi = df1
km = KMeans(n_clusters=5)
km.fit(dfi)
df1['Pred'] = km.predict(dfi)
df1.head(2)  

# Cluster Gender
plt.title("Custers of Annual Income vs Spending Score")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
for i in df1['Pred'].unique():
    dff = df1[df1['Pred']==i]
    plt.scatter(dff['Annual Income (k$)'], dff['Spending Score (1-100)'], label=i)
plt.legend()    
plt.show() 

df.head(2)

df2 = df.drop(['CustomerID','Gender'], axis=1)
df2.head(2)

dfi = df2

# Gender Clusters with Centroids
def kmean_clust(c):
    km = KMeans(n_clusters=c)
    print("Total Clusters =", c)
    km.fit(dfi)
    df2['Pred'] = km.predict(dfi)
    plt.title("Clusters of Annual Income vs Spending Score With Centroids ")
    plt.xlabel("Annual Income")
    plt.ylabel("Spending Score")
    for a in df2['Pred'].unique():
        dff2 = df2[df2['Pred']==a]
        plt.scatter(dff2['Annual Income (k$)'], dff2['Spending Score (1-100)'], label=a)
        # plot the centroids for the clusters
        plt.scatter(km.cluster_centers_[a][1], km.cluster_centers_[a][2], marker="*", s=150, c='k')
    plt.legend()    
    plt.show()   

from ipywidgets import interact
interact(kmean_clust, c=(2,5))

# With Age
dfi = df2.iloc[:,:-1]
dfi.head(2)

def kmean_clust_age(c):
    km = KMeans(n_clusters=c)
    print("Total Clusters =", c)
    km.fit(dfi)
    df2['Prd'] = km.predict(dfi)
    plt.title("Clusters of Age vs Spending Score With Centroids ")
    plt.xlabel("Age")
    plt.ylabel("Spending Score")
    for a in df2['Prd'].unique():
        dff3 = df2[df2['Prd']==a]
        plt.scatter(dff3['Age'], dff3['Spending Score (1-100)'], label=a)
        # plot the centroids for the clusters
        plt.scatter(km.cluster_centers_[a][0], km.cluster_centers_[a][2], marker="*", s=150, c='k')
    plt.legend()    
    plt.show() 

from ipywidgets import interact
interact(kmean_clust_age, c=(2,4))

 Bingo..!!
