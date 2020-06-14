#!/usr/bin/env python
# coding: utf-8

# In[45]:


# This is a small dataset and we will try to only visualise it in different inferring ways
#  Dataset Information: Data is from 22 Jan 2020 to 20 April 2020. It has 9 columns. Here, the final 3 parameters are Cured, Deaths and Confirmed. Out of these 3 
# parameters, accornig to the need, anyone can be considered as the output while building a prediction model. It all depends
# on the requirement, what exactly we want. I have tried to present a general visualization to get a summarized conlusions.


# In[10]:


# import bsic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


# In[4]:


# load the dataset
df = pd.read_csv("d:/covid_19_india.csv")
df.head(2)


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.isna().sum()


# In[44]:


df['Date'].describe() # Data consists of case studies from 82 dates, wkith maxim data of 20th April 2020, as comapred to other
# dates (33 entries)


# In[48]:


df[df['Date']=='20/04/20'].shape


# In[31]:


# State wise number of case studies
x=sb.countplot(df['State/UnionTerritory'])
x.set_xticklabels(df['State/UnionTerritory'].unique(), rotation=90)
plt.show()


# In[17]:


# Maximum case studies are from Kerala State, Minimum case studies are from Nagaland and one unassigned state 


# In[33]:


# Cases of Confirmed Indian Nationals
plt.figure(figsize=(8,6))
x = sb.countplot(df['ConfirmedIndianNational'])
x.set_xticklabels(df['ConfirmedIndianNational'].unique(), rotation=90)
plt.show()


# In[35]:


# It shows that most of the cases do not have the info whether they are indian or foriegners. 
# Over all cases are 1157 out of which unidentified cases on the basis of nationality are 711 so over 82% cases has either
# no information about their nationality provied or it is not documented
# This is the same case with ConfirmedForeignNational


# In[63]:


# In such case, when required, the '-' from ConfirmedIndianNational and ConfirmedForiegnNational can be replaced by Null values
# by the following code
df[['ConfirmedIndianNational','ConfirmedForeignNational']] = df[['ConfirmedIndianNational','ConfirmedForeignNational']].replace(['-','na'], np.nan)
# It can also be rplaces by ther required values.


# In[40]:


# Top 10 States with maximum Deaths
sr_high =df.groupby('State/UnionTerritory')['Deaths'].sum().sort_values(ascending=False).head(10)
plt.bar(sr_high.index,sr_high.values)
plt.title("Top 10 Death-Rate States")
plt.xlabel("States")
plt.xticks(rotation=90)
plt.ylabel("Deaths")
plt.show()


# In[42]:


# Top 15 States with minimum Deaths
sr_high =df.groupby('State/UnionTerritory')['Deaths'].sum().sort_values(ascending=False).tail(15)
plt.bar(sr_high.index,sr_high.values)
plt.title("Top 10 Least Death-Rate States")
plt.xlabel("States")
plt.xticks(rotation=90)
plt.ylabel("Deaths")
plt.show()
# we see that in this list, 14 states have no deaths reported so far


# In[50]:


# Top 10 States with maximum Cured
sr_high =df.groupby('State/UnionTerritory')['Cured'].sum().sort_values(ascending=False).head(10)
plt.bar(sr_high.index,sr_high.values)
plt.title("Top 10 Cured-Rate States")
plt.xlabel("States")
plt.xticks(rotation=90)
plt.ylabel("Cured")
plt.show()


# In[51]:


# Top 15 States with minimum Cured
sr_high =df.groupby('State/UnionTerritory')['Cured'].sum().sort_values(ascending=False).tail(15)
plt.bar(sr_high.index,sr_high.values)
plt.title("Top 10 Least Cured-Rate States")
plt.xlabel("States")
plt.xticks(rotation=90)
plt.ylabel("Cured")
plt.show()


# In[55]:


# Top 10 States with maximum Confirmed Cases
sr_high =df.groupby('State/UnionTerritory')['Confirmed'].sum().sort_values(ascending=False).head(10)
plt.bar(sr_high.index,sr_high.values)
plt.title("Top 10 Confirmed Cases States")
plt.xlabel("States")
plt.xticks(rotation=90)
plt.ylabel("Confirmed")
plt.show()


# In[54]:


# Top 15 States with minimum Confirmed Cases
sr_high =df.groupby('State/UnionTerritory')['Confirmed'].sum().sort_values(ascending=False).tail(15)
plt.bar(sr_high.index,sr_high.values)
plt.title("Top 10 Least Confirmed Cases States")
plt.xlabel("States")
plt.xticks(rotation=90)
plt.ylabel("Confirmed")
plt.show()


# In[59]:


# A combined vsiualization of 'Confirmed','Cured','Deaths' for all the states
df_ccd = df.groupby('State/UnionTerritory')['Confirmed','Cured','Deaths'].sum()
df_ccd.columns


# In[60]:


plt.figure(figsize=(8,6))
plt.plot(df_ccd.index,df_ccd['Confirmed'],label="Confirmed")
plt.plot(df_ccd.index,df_ccd['Cured'],label="Cured")
plt.plot(df_ccd.index,df_ccd['Deaths'],label="Deaths",color='r')
plt.title("Confirmed vs Cured Vs Deaths")
plt.xlabel("States")
plt.xticks(rotation=90)
plt.legend()
plt.show()


# In[64]:


# Let us visualize state wise Indian vs Foreign Cases
# Get the data
df_ct = df[['State/UnionTerritory','ConfirmedIndianNational','ConfirmedForeignNational']]
df_ct.head(2)


# In[65]:


# Drop the null values (the unidentified cases)
df_ct = df_ct.dropna(axis=0)
df_ct['ConfirmedIndianNational'] = df_ct['ConfirmedIndianNational'].astype('int64')
df_ct['ConfirmedForeignNational'] = df_ct['ConfirmedForeignNational'].astype('int64')
type(df_ct)


# In[66]:


# Group them by states and nationality
df_if = df_ct.groupby('State/UnionTerritory')['ConfirmedIndianNational','ConfirmedForeignNational'].sum()
df_if.head()


# In[67]:


# Plot
plt.figure(figsize=(8,6))
plt.plot(df_if.index,df_if['ConfirmedIndianNational'],label="Indian National")
plt.plot(df_if.index,df_if['ConfirmedForeignNational'],label="Foreign National")
plt.title("Confirmed INDIAN vs FORIGNER")
plt.xlabel("States")
plt.ylabel("Cases")
plt.xticks(rotation=90)
plt.legend()
plt.show()


# In[68]:


# Let us visualize state wise confirmed vs Death graph
# Convert dates in datetime format and appropriate format
df['Dates']=pd.to_datetime(df['Date'])
df['Dates'] = df['Dates'].dt.strftime('%d-%m-%y')
df['Month'] = pd.to_datetime(df['Dates']).dt.month


# In[69]:


# Group them
df_mn = df[['Month','Confirmed','Deaths']]
df_mnn = df_mn.groupby('Month')['Confirmed','Deaths'].sum()


# In[70]:


# plot
plt.figure(figsize=(8,6))
plt.plot(df_mnn.index,df_mnn['Confirmed'],label="Confirmed")
plt.plot(df_mnn.index,df_mnn['Deaths'],label="Deaths",color='r')
plt.title("State wiese Confirmed vs Deaths")
plt.xlabel("States")
plt.xticks(rotation=90)
plt.ylabel("Cases")
plt.legend()
plt.show()


# In[ ]:


# Example of building a regression model and then Let's plot the regression


# In[71]:


df = pd.read_csv("d:/covid_19_india.csv")
df.head(2)


# In[72]:


df.info()


# In[74]:


dff = df.drop(['Sno','Time'],axis=1) # not required
dff.head(2)


# In[75]:


from sklearn.preprocessing import LabelEncoder # for encoding
le=LabelEncoder()


# In[76]:


dff['State/UnionTerritory'] = le.fit_transform(dff['State/UnionTerritory']) # encode


# In[263]:


dff['ConfirmedIndianNational'].unique()


# In[77]:


# Just to plot, we have assigned values to null indian and foreign national values. The logic is, if there is some non null
# value in Confirmd cases, we have assigned half to ConfirmedIndianNational and half to ConfirmedForeignNational
for i in range(dff.shape[0]):
    if (dff['ConfirmedIndianNational'][i]=='-'):
        dff['ConfirmedIndianNational'][i] = (dff['Confirmed'][i]/2)
        dff['ConfirmedForeignNational'][i] = (dff['Confirmed'][i]/2)


# In[78]:


dff['ConfirmedIndianNational'].unique()


# In[79]:


dff['Date'] = pd.to_datetime(dff['Date'])
dff['Date'] = dff['Date'].apply(lambda x: x.strftime('%Y%m%d'))


# In[80]:


dff.head(2)


# In[81]:


dfi=dff.iloc[:,dff.columns!='Deaths']
dfo=dff['Deaths']


# In[82]:


from sklearn.linear_model import LinearRegression


# In[83]:


lr=LinearRegression()
lr.fit(dfi,dfo)
lr.score(dfi,dfo)


# In[87]:


dff['Death_lr'] = lr.predict(dfi)
dff.head(2)


# In[88]:


import matplotlib.pyplot as plt


# In[89]:


plt.scatter(dff['Confirmed'],dff['Deaths'],label="Observed")
plt.plot(dff['Confirmed'],dff['Death_lr'],color='r',label="Predicted")
plt.title("Confirmed vs Deaths: Liner Regression")
plt.xlabel("Confirmed Cases")
plt.ylabel("Deaths")
plt.legend()
plt.show()


# In[100]:


def predictDeath(Dt,St,Cn,Cf,Cu,Cnf):
    print("Predicted Deaths :", lr.predict([[Dt,St,Cn,Cf,Cu,Cnf]]))
    plt.scatter(dff['Confirmed'],dff['Deaths'],label="Observed")
    plt.plot(dff['Confirmed'],dff['Death_lr'],color='r',label="Predicted Plot")
    plt.plot(Cnf,lr.predict([[Dt,St,Cn,Cf,Cu,Cnf]]),color='k',marker="*",label="Predicted Case") # example of prediction
    plt.title("Confirmed vs Deaths: Liner Regression")
    plt.xlabel("Confirmed Cases")
    plt.ylabel("Deaths")
    plt.legend()
    plt.show()


# In[101]:


predictDeath(20200431,15,3000,3,0,3003)

