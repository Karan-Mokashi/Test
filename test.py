#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import lifetimes
from datetime import timedelta
import streamlit as st
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title = "Compunnel digital")
st.image("compunnel.png",width=100)

st.title('Customer Lifetime Prediction App')

data= pd.read_csv("CLV_Segmentation.csv", parse_dates=['InvoiceDate'], encoding='unicode_escape')
data= data.drop(['StockCode','Description'], axis=1)
data['Total_Revenue']= data['Quantity'].multiply(data['UnitPrice'])
data= data.dropna(subset=['CustomerID'],axis=0)

#detecting outliers:
q1=data['Quantity'].quantile(0.25)
q2=data['Quantity'].quantile(0.50)
q3=data['Quantity'].quantile(0.75)
iqr=q3-q1
iqr


upper_limit=q3+1.5*iqr
lower_limit=q1-1.5*iqr
upper_limit,lower_limit


#replacing upper values with upper limit and lower values with lower limit
def limit_imputer(value):
    if value > upper_limit:
        return upper_limit
    if value < lower_limit:
        return lower_limit
    else:
        return value

data['Quantity'] = data['Quantity'].apply(limit_imputer)

#detecting outliers:
q1=data['UnitPrice'].quantile(0.25)
q2=data['UnitPrice'].quantile(0.50)
q3=data['UnitPrice'].quantile(0.75)
iqr=q3-q1
iqr


upper_limit=q3+1.5*iqr
lower_limit=q1-1.5*iqr
upper_limit,lower_limit


#replacing upper values with upper limit and lower values with lower limit
def limit_imputer(value):
    if value > upper_limit:
        return upper_limit
    if value < lower_limit:
        return lower_limit
    else:
        return value

data['UnitPrice'] = data['UnitPrice'].apply(limit_imputer)

#detecting outliers:
q1=data['Total_Revenue'].quantile(0.25)
q2=data['Total_Revenue'].quantile(0.50)
q3=data['Total_Revenue'].quantile(0.75)
iqr=q3-q1
iqr


upper_limit=q3+1.5*iqr
lower_limit=q1-1.5*iqr
upper_limit,lower_limit


#replacing upper values with upper limit and lower values with lower limit
def limit_imputer(value):
    if value > upper_limit:
        return upper_limit
    if value < lower_limit:
        return lower_limit
    else:
        return value

data['Total_Revenue'] = data['Total_Revenue'].apply(limit_imputer)

#pip freeze > requirements.txt

# Transforming the data to customer level for the analysis
customer = data.groupby('CustomerID').agg({'InvoiceDate':lambda x: (x.max() - x.min()).days, 
                                                   'InvoiceNo': lambda x: len(x),
                                                  'Total_Revenue': lambda x: sum(x)})

customer.columns = ['Age', 'Frequency', 'Total_Revenue']


# Calculating the necessary variables for CLV calculation
Average_revenue = round(np.mean(customer['Total_Revenue']),2)
#print(f"Average revenue: ${Average_revenue}")

Purchase_freq = round(np.mean(customer['Frequency']), 2)
#print(f"Purchase Frequency: {Purchase_freq}")

Retention_rate = customer[customer['Frequency']>1].shape[0]/customer.shape[0]
churn = round(1 - Retention_rate, 2)


Profit_margin = 0.05 
CLV = round(((Average_revenue * Purchase_freq/churn)) * Profit_margin, 2)


#Cohort Analysis
# Transforming the data to customer level for the analysis
customer = data.groupby('CustomerID').agg({'InvoiceDate':lambda x: x.min().month, 
                                                   'InvoiceNo': lambda x: len(x),
                                                  'Total_Revenue': lambda x: np.sum(x)})
customer.columns = ['Start_Month', 'Frequency', 'Total_Revenue']

# Calculating CLV for each cohort
months = ['Jan', 'Feb', 'March', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
Monthly_CLV = []

for i in range(1, 13):
    customer_m = customer[customer['Start_Month']==i]
    Average_revenue = round(np.mean(customer_m['Total_Revenue']),2)
    Purchase_freq = round(np.mean(customer_m['Frequency']), 2)
    Retention_rate = customer_m[customer_m['Frequency']>1].shape[0]/customer_m.shape[0]
    churn = round(1 - Retention_rate, 2)
    CLV = round(((Average_revenue * Purchase_freq/churn)) * Profit_margin, 2)
    Monthly_CLV.append(CLV)
    
monthly_clv = pd.DataFrame(zip(months, Monthly_CLV), columns=['Months', 'CLV'])    


# Creating the summary data using summary_data_from_transaction_data function
summary = lifetimes.utils.summary_data_from_transaction_data(data, 'CustomerID', 'InvoiceDate', 'Total_Revenue' )
summary = summary.reset_index()

#NBD Model
# Create a distribution of frequency to understand the customer frequence level
summary['frequency'].plot(kind='hist', bins=50)
#print(summary['frequency'].describe())
#print("---------------------------------------")
one_time_buyers = round(sum(summary['frequency'] == 0)/float(len(summary))*(100),2)


# Fitting the BG/NBD model
bgf = lifetimes.BetaGeoFitter(penalizer_coef=0.0)
bgf.fit(summary['frequency'], summary['recency'], summary['T'])

# Compute the customer alive probability
summary['probability_alive'] = bgf.conditional_probability_alive(summary['frequency'], summary['recency'], summary['T'])

#Predict future transaction for the next 30 days based on historical dataa
t = 30
summary['pred_num_txn'] = round(bgf.conditional_expected_number_of_purchases_up_to_time(t, summary['frequency'], summary['recency'], summary['T']),2)
summary.sort_values(by='pred_num_txn', ascending=False).head(10).reset_index()


# Checking the relationship between frequency and monetary_value
return_customers_summary = summary[summary['frequency']>0]
#print(return_customers_summary.shape)


# Modeling the monetary value using Gamma-Gamma Model
ggf = lifetimes.GammaGammaFitter(penalizer_coef=0.001)
ggf.fit(return_customers_summary['frequency'],
       return_customers_summary['monetary_value'])


# Calculating the conditional expected average profit for each customer per transaction
summary = summary[summary['monetary_value'] >0]
summary['exp_avg_revenue'] = ggf.conditional_expected_average_profit(summary['frequency'],
                                       summary['monetary_value'])


n_month=int(st.text_input("input CLV Prediction in Month",1))


# Predicting Customer Lifetime Value for the next 30 days
summary['predicted_clv'] =  ggf.customer_lifetime_value(bgf,
                                                               summary['frequency'],
                                                               summary['recency'],
                                                               summary['T'],
                                                               summary['monetary_value'],
                                                               time=n_month,     # lifetime in months
                                                               freq='D',   # frequency in which the data is present(T)      
                                                               discount_rate=0.01) # discount rate


# manually calculate
summary['manual_predicted_clv'] = summary['pred_num_txn'] * summary['exp_avg_revenue']

# CLV in terms of profit (profit margin is 5%)
profit_margin = 0.05
summary['CLV'] = summary['predicted_clv'] * profit_margin
st.write(summary)
st.write('MAE')
st.write(mean_absolute_error(summary['manual_predicted_clv'],summary['predicted_clv']))

st.write(sns.scatterplot(x='manual_predicted_clv',y='predicted_clv',data=summary)



data['Hour'] = data['InvoiceDate'].dt.hour
data['Weekday'] = data['InvoiceDate'].dt.weekday
#data_uk['WeekdayName'] = data['InvoiceDate'].dt.weekday_name
data['Month'] = data['InvoiceDate'].dt.month

#RFM Modelling

def RFM_Features(df, customerID, invoiceDate, transID, revenue):
    ''' Create the Recency, Frequency, and Monetary features from the data '''
    # Final date in the data + 1 to create latest date
    latest_date = df[invoiceDate].max() + timedelta(1)
    
    # RFM feature creation
    RFMScores = df.groupby(customerID).agg({invoiceDate: lambda x: (latest_date - x.max()).days, 
                                          transID: lambda x: len(x), 
                                          revenue: lambda x: sum(x)})
    
    # Converting invoiceDate to int since this contains number of days
    RFMScores[invoiceDate] = RFMScores[invoiceDate].astype(int)
    
    # Renaming column names to Recency, Frequency and Monetary
    RFMScores.rename(columns={invoiceDate: 'Recency', 
                         transID: 'Frequency', 
                         revenue: 'Monetary'}, inplace=True)
    
    return RFMScores.reset_index()


RFM = RFM_Features(df=data, customerID= "CustomerID", invoiceDate = "InvoiceDate", transID= "InvoiceNo", revenue="Total_Revenue")

# Creating quantiles 
Quantiles = RFM[['Recency', 'Frequency', 'Monetary']].quantile([0.25, 0.50, 0.75])
Quantiles = Quantiles.to_dict()


# Creating RFM ranks
def RFMRanking(x, variable, quantile_dict):
    ''' Ranking the Recency, Frequency, and Monetary features based on quantile values '''
    
    # checking if the feature to rank is Recency
    if variable == 'Recency':
        if x <= quantile_dict[variable][0.25]:
            return 4
        elif (x > quantile_dict[variable][0.25]) & (x <= quantile_dict[variable][0.5]):
            return 3
        elif (x > quantile_dict[variable][0.5]) & (x <= quantile_dict[variable][0.75]):
            return 2
        else:
            return 1
    
    # checking if the feature to rank is Frequency and Monetary
    if variable in ('Frequency','Monetary'):
        if x <= quantile_dict[variable][0.25]:
            return 1
        elif (x > quantile_dict[variable][0.25]) & (x <= quantile_dict[variable][0.5]):
            return 2
        elif (x > quantile_dict[variable][0.5]) & (x <= quantile_dict[variable][0.75]):
            return 3
        else:
            return 4

RFM['R'] = RFM['Recency'].apply(lambda x: RFMRanking(x, variable='Recency', quantile_dict=Quantiles))
RFM['F'] = RFM['Frequency'].apply(lambda x: RFMRanking(x, variable='Frequency', quantile_dict=Quantiles))
RFM['M'] = RFM['Monetary'].apply(lambda x: RFMRanking(x, variable='Monetary', quantile_dict=Quantiles))


RFM['Group'] = RFM['R'].apply(str) + RFM['F'].apply(str) + RFM['M'].apply(str)

RFM["Score"] = RFM[['R', 'F', 'M']].sum(axis=1)


# Loyalty levels
loyalty = ['Bronze', 'Silver', 'Gold', 'Platinum']
RFM['Loyalty_Level'] = pd.qcut(RFM['Score'], q=4, labels= loyalty)


