#!/usr/bin/env python
# coding: utf-8

# # Stock Price Prediction - The Right Way Approach: Using Percentage Change in Close Price¶

# # Data Preprocessing

# In[ ]:


# Import All the Important Libraries


# In[1]:


# for sagemaker and iam role
import boto3 # AWS Python SDK
from sagemaker import get_execution_role
role = get_execution_role()

# for tensorflow libraries and modules through sagemaker
import sagemaker.tensorflow 
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.models import Model
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# for normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# for arrays and dataframes
import datetime 
import pandas as pd
import numpy as np

# for plotting and visualization
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[ ]:


# Import data saved from AWS Data Exchange to S3 bucket


# In[2]:


my_bucket = 'stock-price-predictor' # declare bucket name
my_file = 'fred-SP500/dataset/fred-sp500.csv' # declare file path with S3 bucket

# file
data_location = 's3://stock-price-predictor/fred-SP500/dataset/fred-sp500.csv'.format(my_bucket,my_file)
data = pd.read_csv(data_location)
data.tail()


# In[ ]:


# data cleaning and setting for appropriate formats


# In[3]:


# change 'DATE', 'SP500' format and relabel column names
# rename and convert date column to datetime
data['DATE'] = pd.to_datetime(data['DATE'])
data.rename(columns={'DATE': 'date'}, inplace=True)
# rename and convert closing price column from string to float
data.rename(columns={'SP500': 'close'}, inplace=True)
data['close'] = pd.to_numeric(data['close'], errors='coerce')

# remove all rows with missing close price values
data.dropna(subset=['close'], inplace=True)
# set date variable as index
data.set_index('date', inplace=True)
data.sort_index()
data.tail()


# In[4]:


# plot historical data
data.reset_index().plot(x='date',y='close', figsize=(16,6))
plt.xlabel('date')
plt.ylabel('Close Price')
plt.title('S&P500 Index Historical Chart')


# In[ ]:


# calculating EMA and MACD on entire dataset with rolling windows


# In[5]:


ema1 = data.close.ewm(span=12, adjust=False).mean()
ema2 = data.close.ewm(span=26, adjust=False).mean()
macd = ema1 - ema2
ema3 = macd.ewm(span=9, adjust=False).mean()
plt.figure(figsize=(16,6))
plt.plot(data.index, macd, label='S&P MACD', color = 'purple')
plt.plot(data.index, ema3, label='Signal Line', color='brown')
plt.legend(loc='upper left')
plt.show()


# In[ ]:


# calculating EMA and MACD since 1-Jan-2020

#data1 = pd.DataFrame(data['2020-01-01':])
#ema1 = data1.close.ewm(span=12, adjust=False).mean()
#ema2 = data1.close.ewm(span=26, adjust=False).mean()
#macd = ema1 - ema2
#ema3 = macd.ewm(span=9, adjust=False).mean()
#plt.figure(figsize=(16,6))
#plt.plot(data1.index, macd, label='S&P MACD', color = 'purple')
#plt.plot(data1.index, ema3, label='Signal Line', color='brown')
#plt.legend(loc='upper left')
#plt.show()


# # Feature Engineering - Log Returns

# In[ ]:


# calculate the percentage change in close price
# this has the benefit of normalization. also, returns have the benefit of stationarity
# with better statistical properties to measure over time


# In[9]:


#pct variable creation
data['PrevClose'] = data['close'].shift(1)
data['returns'] = (data['close'] - data['PrevClose']) / data['PrevClose']
data.head()


# In[10]:


#plot of log returns: centered around mean zero
plt.figure(1, figsize=(16,6))
plt.plot(data['returns'])


# In[ ]:


# We next normalize our data by taking the log of the pct returns variable


# In[11]:


# Log Normality
data['log_returns'] = np.log(1 + data['returns'])
data.head()


# In[10]:


#plot of log returns: centered around mean zero
plt.figure(1, figsize=(16,6))
plt.plot(data['log_returns'])


# In[12]:


#drop missing values and retain only close price and log returns variables
data.dropna(inplace=True)
data = data[['close', 'log_returns']]
data_new = data.values
data_new


# # Train Test Split

# In[ ]:


# splitting our dataset into test and learn


# In[13]:


# train set
training_set = pd.DataFrame(data[:'2020-12-31'])
training_set_len = len(data[:'2020-12-31'])
training_set_len


# In[14]:


# test set
test_set = pd.DataFrame(data['2021-01-01':])
test_set_len = len(data['2021-01-01':])
test_set_len


# In[15]:


# feature scaling (on close price)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(training_set)
scaled_data[:5]


# In[16]:


# Setting the actual normalized close price equal to y
y = scaled_data[:,[0]]
y[:5]


# In[17]:


# train set
x_train = scaled_data[0:training_set_len, :]
y_train = y[:training_set_len]
y_train.shape


# In[17]:


print(x_train.shape)
print(y_train.shape)


# In[18]:


#setting the real close price for plotting validation
real_stock_price = test_set.close.values

#transforming the test data before prediction
inputs = data[len(data) - test_set_len - 60:].values
inputs = inputs
inputs = scaler.transform(inputs)
y_test = inputs[:,[0]]
y_test[:5]


# In[19]:


assert len(x_train) == len(y_train)
assert len(inputs) == len(y_test)


# In[20]:


print(inputs.shape)
print(y_test.shape)


# # Labelling

# In[ ]:


# preparing the train and test data 


# In[19]:


# splitting data into buckets of 60 timestamps for prediction
n = 60
Xtrain = []
ytrain = []
Xtest = []
ytest = []
for i in range(n, len(x_train)):
    Xtrain.append(x_train[i - n : i, : x_train.shape[1]])
    ytrain.append(y_train[i])  # predict next record
for i in range(n, len(inputs)):
    Xtest.append(inputs[i - n : i, : inputs.shape[1]])
    ytest.append(y_test[i])  # predict next record


# In[20]:


Xtrain, ytrain = (np.array(Xtrain), np.array(ytrain))
Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], Xtrain.shape[2]))

Xtest, ytest = (np.array(Xtest), np.array(ytest))
Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], Xtest.shape[2]))


# In[23]:


print(Xtrain.shape)
print(ytrain.shape)
print("---")
print(Xtest.shape)
print(ytest.shape)


# # Model

# In[33]:


#Initializing the RNN

model = Sequential()

#Adding the first LSTM layer and some Dropout Regularization
model.add(LSTM(units = 50, return_sequences = True, input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
model.add(Dropout(0.2))
              
#Adding a second LSTM layer and some Dropout Regularization
model.add(LSTM(units = 50, return_sequences = False))
model.add(Dropout(0.2))
              
#Adding the output layer
model.add(Dense(25))
model.add(Dense(1))


# In[34]:


#Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Fitting the RNN to the Training Set
model.fit(Xtrain, ytrain, epochs = 20, batch_size = 30, validation_data=(Xtest, ytest), verbose=1 )


# In[51]:


# save model
model.save('my_pct_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model


# In[4]:


model = load_model('my_pct_model.h5')


# In[5]:


model.summary()


# In[21]:


# Model Prediction

trainPredict = model.predict(Xtrain)
testPredict = model.predict(Xtest)


# In[22]:


# updating the shape of the model
trainPredict = np.c_[trainPredict, np.zeros(trainPredict.shape)]
testPredict = np.c_[testPredict, np.zeros(testPredict.shape)]


# In[23]:


# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainPredict = [x[0] for x in trainPredict]

testPredict = scaler.inverse_transform(testPredict)
testPredict = [x[0] for x in testPredict]


# In[24]:


print(trainPredict[:5])
print(testPredict[:5])


# In[25]:


#checking the root mean square error
rmse = np.sqrt(np.mean(testPredict - real_stock_price)**2)
rmse


# In[ ]:


# Building the test set for validation for stock price till YTD 15-Apr-2021


# In[ ]:


# plot the predicted v actual stock price till YTD 15-Apr-2021


# In[26]:


plt.figure(figsize=(20,10))
plt.plot(real_stock_price, color = 'green', label = 'S&P500 Stock Price')
plt.plot(testPredict, color = 'red', label = 'Predicted S&P500 Stock Price')
plt.title('S&P500 Stock Price Prediction')
plt.xlabel('Trading Day')
plt.ylabel('S&P500 Stock Price')
plt.legend()
plt.show()


# # Forecast

# In[ ]:


# Predicting the next date (16-Apr-2021) stock price not in series


# In[27]:


# Single-step forecast

#use the last 60 days to predict the next one into the future
#get the last 60 days close price and convert the dataframe to an array

last_60_days = data[-60:].values

#scale the data to be values between 0 and 1
last_60_days = scaler.transform(last_60_days)

#create an empty list
x_test = []
#append the last 60 days prices
x_test.append(last_60_days)

#convert the x_test to a numpy array
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

#get the predicted scaled price
pred_price = model.predict(x_test)
pred_price = np.c_[pred_price, np.zeros(pred_price.shape)]

#undo the scaling
pred_price = scaler.inverse_transform(pred_price)
pred_price = pred_price[:1,0]
print(pred_price)


# In[28]:


#actual close price (16-Apr-2021) = 4185.47
#diff
print(pred_price - 4185.47)


# In[ ]:


# Predicting the next 6 days (16-Apr - 23-Apr -2021) stock price not in series


# In[29]:


# Multi-step forecast

#use the last 60 days to predict the next one into the future
#get the last 60 days close price and convert the dataframe to an array

y = data[['close']]
newr = {'close' : [4122.09919687, 4128.57390401, 4132.87211125, 4135.53667668, 4136.87386856]}
y = y.append(pd.DataFrame(newr), ignore_index=True)
y['PrevClose'] = y['close'].shift(1)
y['returns'] = (y['close'] - y['PrevClose']) / y['PrevClose']
y['log_returns'] = np.log(1 + y['returns'])
y = y[['close', 'log_returns']]
last_y = y[-n:]

#scale the data to be values between 0 and 1
last_y_d = scaler.transform(last_y)

#create an empty list
x_test1 = []
#append the last 60 days prices
x_test1.append(last_y_d)

#convert the x_test to a numpy array
x_test1 = np.array(x_test1)
x_test1 = np.reshape(x_test1, (x_test1.shape[0], x_test1.shape[1], x_test1.shape[2]))

#get the predicted scaled price
pred_price1 = model.predict(x_test1)
pred_price1 = np.c_[pred_price1, np.zeros(pred_price1.shape)]

#undo the scaling
pred_price1 = scaler.inverse_transform(pred_price1)
pred_price1 = pred_price1[:1,0]
print(pred_price1)


# In[30]:


validation_target = pd.to_numeric(['4185.47', '4163.26', '4134.94', '4173.42', '4134.98', '4180.17'], errors='coerce')
validation_predictions = pd.to_numeric(['4122.09919687', '4128.57390401', '4132.87211125', '4135.53667668', '4136.87386856', '4137.14878153'], errors='coerce')


# In[31]:


#checking the root mean square error
rmse = np.sqrt(np.mean(validation_predictions - validation_target)**2)
rmse


# In[ ]:


# Plotting Real v Predicted - Multi-step forecast


# In[33]:


plt.figure(figsize=(20,10))
plt.plot(validation_target, color = 'green', label = 'S&P500 Stock Price')
plt.plot(validation_predictions, color = 'red', label = 'Predicted S&P500 Stock Price')
plt.title('S&P500 Stock Price Prediction')
plt.xlabel('Trading Day')
plt.ylabel('S&P500 Stock Price')
plt.legend()
plt.show()


# # Thank You!
