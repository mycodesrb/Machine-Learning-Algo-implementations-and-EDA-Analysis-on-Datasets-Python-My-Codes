#!/usr/bin/env python
# coding: utf-8

# In[21]:


# Here, I have tried to visualize the data in an example of timeseries i.e, adjusting the visualization in fixed interval of
# time. Tieseries helps us to understand the pattern, trend, seasonality etc about the data. Lets execute a small example.
# Here I will plot Observed vs Predicted deaths of the country (Overall)

# About Dataset: Dataset contains data of 185 countries with 8 columns Viz, Date, Country/Region, Province/State, Lat and Long,
# Confirmed, Recovered and Deaths


# In[22]:


import pandas as pd
import numpy as np


# In[27]:


# Load the data
df =pd.read_csv("d:/covid_19_time_series.csv")
df.head()


# In[28]:


df.isnull().sum() # Checking null values


# In[29]:


df.shape


# In[30]:


df.info()


# In[31]:


# We see NaN values in Province/State, lets handle it
df['Province/State'] = df['Province/State'].fillna('NA')
df[df['Province/State']=='NA'].head(1)


# In[32]:


df['Confirmed'] = df['Confirmed'].fillna(0.0) #fill NA with 0
df['Confirmed'].isnull().sum()


# In[33]:


df['Recovered'] = df['Recovered'].fillna(0.0) #fill NA with 0
df['Recovered'].isnull().sum()


# In[34]:


df['Deaths'] = df['Deaths'].fillna(0.0) #fill NA with 0
df['Deaths'].isnull().sum()


# In[35]:


df.info()


# In[36]:


df['Month'] = pd.to_datetime(df['Date']).dt.month
df['Day'] = pd.to_datetime(df['Date']).dt.day


# In[37]:


df.head(2)


# In[38]:


import matplotlib.pyplot as plt


# In[39]:


df.groupby('Date')['Deaths'].sum()


# In[40]:


df.info()


# In[41]:


df.head(2)


# In[42]:


df['Date'].describe() 


# In[43]:


df['Month'].unique()


# In[52]:


# Plot country wise Month wise DAILY data (here DAILY data is or MONTH is the repeated interval)
def plotdaily(month,country):
    dff = df[(df['Month']==month) & (df['Country/Region']==country)]
    sr = dff.groupby('Day')['Confirmed','Recovered','Deaths'].sum()
    x = [str(i) for i in sr.index]
    plt.plot(x,sr['Confirmed'], label="Confirmed")
    plt.plot(x,sr['Recovered'],color='g',label='Recovered')
    plt.plot(x,sr['Deaths'],color='r',label='Deaths')
    plt.title("Country wise Month wise DAILY DATA")
    plt.xlabel("Month's Day")
    plt.ylabel("Cases")
    plt.legend()
    plt.show()


# In[48]:


from ipywidgets import interact


# In[49]:


interact(plotdaily, month=df['Month'].unique(), country=df['Country/Region'].unique())


# In[53]:


# PLot MONTH wise country wise data
def plotmonthly(country):
    df_mn = df[df['Country/Region']==country]
    sr = df_mn.groupby('Month')['Confirmed','Recovered','Deaths'].sum()
    x = [str(i) for i in sr.index]
    plt.plot(x,sr['Confirmed'], label="Confirmed")
    plt.plot(x,sr['Recovered'],color='g',label='Recovered')
    plt.plot(x,sr['Deaths'],color='r',label='Deaths')
    plt.title("Country wise Month wise DATA")
    plt.xlabel("Month")
    plt.ylabel("Cases")
    plt.legend()
    plt.show()


# In[54]:


interact(plotmonthly, country=df['Country/Region'].unique())


# In[58]:


# Lets plot entire data date wise with observed data vs predicted data
df.head(2)


# In[59]:


data = df
df.info()


# In[60]:


from sklearn.preprocessing import LabelEncoder


# In[61]:


data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].apply(lambda x: x.strftime('%Y%m%d'))


# In[62]:


data.info()


# In[63]:


le=LabelEncoder()


# In[64]:


data['Country/Region'] =le.fit_transform(data['Country/Region'])
data['Province/State'] =le.fit_transform(data['Province/State'])


# In[65]:


data = data.drop(['Day','Month'], axis=1)
data.head(2)


# In[66]:


dfi=data.iloc[:,:-1]
dfo=data['Deaths']


# In[67]:


from sklearn.linear_model import LinearRegression


# In[68]:


lr= LinearRegression()
lr.fit(dfi,dfo)
lr.score(dfi,dfo)


# In[69]:


data['Pred_lr'] = lr.predict(dfi)
data.head()


# In[71]:


# Test Plot: for a country, date wise deaths, observed and predicted
dfc = data[data['Country/Region']==1]
plt.scatter(dfc['Date'],dfc['Deaths'], label='Actual')
plt.plot(dfc['Date'],dfc['Pred_lr'],color='r', label='Predicted')
plt.title("Observed and Predicted Deaths of a country DATE wise")
plt.xlabel("Dates")
plt.ylabel("Cases")
plt.xticks(rotation=90)
plt.rcParams['figure.figsize'][0]=20
plt.legend()
plt.show()


# In[ ]:


# Likewise we can plot data for each country on DAY wise, Month wise i.e, repeated and fixed time interval:

