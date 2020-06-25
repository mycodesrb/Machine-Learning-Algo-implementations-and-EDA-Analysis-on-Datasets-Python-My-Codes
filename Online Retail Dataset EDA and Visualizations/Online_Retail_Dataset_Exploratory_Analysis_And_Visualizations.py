#!/usr/bin/env python
# coding: utf-8

# In[328]:


# Here I have done EDA on Online Retail Dataset. I actual, this dataset has combined data from 3 countries and it has over
# half a million records. So, for fast processing, i lsiced it and done the EDA on around 28k+ records. It has records from all 
# 3 countries.

# Dataset Information: Dataset is about the inventory of a Retal Store which has 8 Columns. 
# Invoice: the invoice number
# StockCode: the code for Item/Product
# Description: Info about the item/Product
# Quantity: Number of items purchased against that Invoice
# Invoice Date: Date of Purchase
# UnitPrice: Unit Price of Item/Product
# CustomerID: Identification ID for Customer
# Country


# In[329]:


# Import the basic libraries/Packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


# In[330]:


# Load Data
df = pd.read_excel("d:/Online_Retail_Sliced.xlsx")
df.head(2)


# In[331]:


df.shape


# In[332]:


# Let us add one column and consider it as a Target Column. 
df['Sales'] = df['Quantity']*df['UnitPrice']
df.head(2)


# In[333]:


df.info()


# In[334]:


df.isnull().sum() # We have few nulls in Description and 2725 in CustomerID. Let's Clean them


# In[335]:


df[df['Description'].isnull()].head() # way to see the null values


# In[336]:


# Replace all null decriptions with " NO Description Available" text
nd = "No Description Available"
df['Description'] = df['Description'].fillna(nd)
print(len(df[df['Description'].isnull()])) #Confirm for no null values

#crosscheck
df[df['InvoiceNo']==536414] # Shows the changed Description


# In[337]:


# CustomerID: We have almost 3k Null records and we can not delete them as they have made transaction so we will replace null
# with 0.
df[df['CustomerID'].isnull()].head()


# In[338]:


df['CustomerID'] = df['CustomerID'].fillna(0.0) # Fill NaN with 0
print(len(df[df['CustomerID'].isnull()])) # Check for NaN now
#Crosscheck
df[df['InvoiceNo']==536414] #Done


# In[339]:


df.isnull().sum() # No more Null Values


# In[340]:


# Analysis Starts
len(df['InvoiceNo'].unique()) # 1553 Unique purchases were made where items were purchased


# In[341]:


lsinv = df['InvoiceNo'].value_counts().head(10) # Top Invoices with maximum Products out of all records
x = [str(i) for i in lsinv.index]
plt.barh(x, lsinv.values)
plt.title("Top Invoices for Maximum Products Purchased")
plt.xlabel("Quantity")
plt.ylabel("Invoice Numbers")
plt.show()


# In[342]:


len(df['StockCode'].unique()) # Total 2766 Products are there in the Retail Store


# In[343]:


# Top selling products w.r.t Sales
srpr= df.groupby('StockCode')['Sales'].sum().sort_values(ascending=False).head(10)
srpr


# In[344]:


x = [str(i) for i in srpr.index]
plt.bar(x, srpr.values)
plt.title("Top Selling Products Overall")
plt.xlabel("Product Codes")
plt.ylabel("Sales")
plt.xticks(rotation=90)
plt.show()


# In[345]:


print("Best Sold Item :", srpr[0:1].index.values)
t = df.groupby('StockCode')['Sales'].sum().sort_values(ascending=False)
print('Least Sold Item :', t.tail(1).index.values) 


# In[346]:


t = df.groupby('StockCode')['Sales'].sum().sort_values(ascending=False)
t.tail(1)


# In[347]:


# Top and Least selling products w.r.t Quantity
srqt = df.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False).head()
srqt


# In[348]:


x = [str(i) for i in srqt.index]
plt.bar(x, srqt.values)
plt.title("Top Selling Products w.r.t. Quantities")
plt.xlabel("Product Codes")
plt.ylabel("Quantity")
plt.xticks(rotation=90)
plt.show()


# In[349]:


srqtt = df.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False)
print("Least Sold Item w.r.t Quantity is :", srqtt.tail(1).index.values)


# In[350]:


df['InvoiceDate'].describe()


# In[351]:


# Data is from 1st December 2010 to 9th December 2011. MAximum Purchases are done on 3rd of December 2010 with 593 times
# purchases 


# In[352]:


sb.countplot(df['Country'])
# Most sales are done in UK country, followed by German and then France


# In[371]:


plt.hist(df['Sales'], bins=[-300,-200,-100,0,100,200,300])
plt.show()
# Most of the sales values are there between 0 to 100 (more than 25k values) 


# In[373]:


# If we further break the above histogram, we get more clear idea
plt.hist(df['Sales'], bins=[0,10,20,30,40,50,60])
plt.show()
# We see that around 12000 values are there for sales amount of 10 to 20 (Maximum) and least are from 40 to 60.


# In[374]:


# Below plot shows the presence of outliers
sb.boxplot(df['Sales'])
plt.show()


# In[375]:


df['Sales'].describe()
# Minimum value is approx -8322 and max is 4161. Mean is quite low as comapred to min/max values which shows data is quite
# scattered. It shows the presence of outliers, which we already checked in the boxplot.


# In[353]:


df.head(2)


# In[354]:


# Top selling items Country Wise
for c in df['Country'].unique():
    dfc = df[df['Country']==c]
    sru = dfc.groupby('StockCode')['Sales'].sum().sort_values(ascending=False).head()
    x = [str(i) for i in sru.index]
    plt.title("Top Selling Items w.r.t Sales")
    plt.xlabel("Product Codes")
    plt.ylabel("Sales")
    plt.bar(x, sru.values, label=c)
    plt.legend()
    plt.show()


# In[355]:


# Least selling items Country Wise
for c in df['Country'].unique():
    dfc = df[df['Country']==c]
    sru = dfc.groupby('StockCode')['Sales'].sum().sort_values(ascending=False).tail()
    x = [str(i) for i in sru.index]
    plt.title("Least Selling Items w.r.t Sales")
    plt.xlabel("Product Codes")
    plt.ylabel("Sales")
    plt.bar(x, sru.values, label=c)
    plt.legend()
    plt.show()
    # Here Negative values shows the retirns post purchase


# In[356]:


# To analyse data on the basis of Days and months, we need to extract these values in the dataframe
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day
df.head(2)


# In[376]:


# Country wise Month wise sales
for c in df['Country'].unique():
    dfc = df[df['Country']==c]
    srmx = dfc.groupby('Month')['Sales'].sum()
    x = [str(i) for i in srmx.index]
    plt.title("Country Wise Month Wise Total Sales")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.plot(x, srmx.values, "--go",label=c)
    plt.legend()
    plt.show()
    # UK shows Sale only in the month of December 
    # Germany shos Least Sale in February and Maximum Sale in October
    # France Shows least Sale in April and Maximum Sale in November


# In[358]:


# Country Wise Month wise Day wise Sales
def cmd(country,month):
        if country=='United Kingdom':
            print('UK has only December Data')
            month=12
        dfd = df[(df['Country']==country) & (df['Month']==month)]
        s = dfd.groupby('Day')['Sales'].sum()
        sb.barplot(x=s.index, y=s.values, ci=None)
        plt.show()


# In[359]:


from ipywidgets import interact
interact(cmd, country=df['Country'].unique(), month=df['Month'].unique())


# In[360]:


df.head(2)


# In[361]:


df['Hour'] = df['InvoiceDate'].dt.hour
df.head(2)


# In[362]:


# Country Wise Month wise Month wise Hourly Sales
def cmd(country,month):
        if country=='United Kingdom':
            print('UK has only December Data')
            month=12
        dfd = df[(df['Country']==country) & (df['Month']==month)]
        s = dfd.groupby('Hour')['Sales'].sum().sort_index()
        sb.barplot(x=s.index, y=s.values, ci=None)
        plt.show()


# In[363]:


from ipywidgets import interact
interact(cmd, country=df['Country'].unique(), month=df['Month'].unique())
# Here we can see in Each month, what was the hour where maximum business was occured, even we can see the hour where the 
# business was lowest


# In[377]:


# Which days of week maximum sales occur?
df.head(2)


# In[381]:


ddays={'MONDAY':0,'TUESDAY':1,'WEDNESDAY':2,'THURSDAY':3,'FRIDAY':4,'SATURDAY':5,'SUNDAY':6}


# In[378]:


df['Weekday']=df['InvoiceDate'].dt.dayofweek
df.head(2)


# In[393]:


sr = df.groupby('Weekday')['Sales'].sum().sort_values(ascending=False)
print("Weekday wise Sales, Sorted by Sales Amount :")
print("---------------------------------------------")
j=0
for i in sr.index:
    for dk,dv in ddays.items():
        if dv==i:
            print(dk+"'S Sales: ", round(sr.values[j]))
            print("--------------------------------")    
            j+=1


# In[ ]:


# Like these, there can be many more calculations and visualizations like repeate customers, clusters of Royal and normal 
# customers etc. Depending upn the need, tasks can be made.

