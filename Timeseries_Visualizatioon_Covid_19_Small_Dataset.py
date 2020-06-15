#!/usr/bin/env python
# coding: utf-8

# Here, I have tried to visualize the data in an example of timeseries i.e, adjusting the visualization in fixed interval of
# time. Tieseries helps us to understand the pattern, trend, seasonality etc about the data. Lets execute a small example.
# Here I will plot Observed vs Predicted deaths of the country (Overall)

# About Dataset: Dataset contains data of 185 countries with 8 columns Viz, Date, Country/Region, Province/State, Lat and Long,
# Confirmed, Recovered and Deaths

import pandas as pd
import numpy as np

# Load the data
df =pd.read_csv("d:/covid_19_time_series.csv")
df.head()

df.isnull().sum() # Checking null values

df.shape

df.info()

# We see NaN values in Province/State, lets handle it
df['Province/State'] = df['Province/State'].fillna('NA')
df[df['Province/State']=='NA'].head(1)

df['Confirmed'] = df['Confirmed'].fillna(0.0) #fill NA with 0
df['Confirmed'].isnull().sum()

df['Recovered'] = df['Recovered'].fillna(0.0) #fill NA with 0
df['Recovered'].isnull().sum()

df['Deaths'] = df['Deaths'].fillna(0.0) #fill NA with 0
df['Deaths'].isnull().sum()

df.info()

df['Month'] = pd.to_datetime(df['Date']).dt.month
df['Day'] = pd.to_datetime(df['Date']).dt.day

df.head(2)

import matplotlib.pyplot as plt

df.groupby('Date')['Deaths'].sum()

df.info()

df.head(2)

df['Date'].describe() 

df['Month'].unique()

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

from ipywidgets import interact
interact(plotdaily, month=df['Month'].unique(), country=df['Country/Region'].unique())

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

interact(plotmonthly, country=df['Country/Region'].unique())

# Lets plot entire data date wise with observed data vs predicted data
df.head(2)

data = df
df.info()

from sklearn.preprocessing import LabelEncoder

data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].apply(lambda x: x.strftime('%Y%m%d'))

data.info()

le=LabelEncoder()

data['Country/Region'] =le.fit_transform(data['Country/Region'])
data['Province/State'] =le.fit_transform(data['Province/State'])

data = data.drop(['Day','Month'], axis=1)
data.head(2)

dfi=data.iloc[:,:-1]
dfo=data['Deaths']

from sklearn.linear_model import LinearRegression

lr= LinearRegression()
lr.fit(dfi,dfo)
lr.score(dfi,dfo)

data['Pred_lr'] = lr.predict(dfi)
data.head()

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

# Likewise we can plot data for each country on DAY wise, Month wise i.e, repeated and fixed time interval:

