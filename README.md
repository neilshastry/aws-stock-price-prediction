# Stock Price Prediction with LSTM in AWS Sagemaker on the Cloud

## Background
This is a fun personal project where I wanted to explore some of the math, technical indicators and use deep learning in a cloud native environment to validate the growing hypothesis that machine learning can predict stock prices. For this illustration, I have used the broad market S&P 500 index - however, this project can be readily extended to evaluate other individual or a basket of stocks.

## Objectives
Evaluate time series data of S&P 500 Close price through LSTM RNN and explore prediction feasibility for future close price.

## Tools
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)

## Prerequisite
Sign-up to AWS free-tier through the console to access the services described in this project.

[<img width="1414" alt="AWS Sign-up" src="https://user-images.githubusercontent.com/36125669/115541657-fcea2500-a2d1-11eb-9054-bc411e8b49ba.png">](https://aws.amazon.com/free/)

**Note:** This project can also be built on a local anaconda distribution through Jupyter or another cloud provider. We have used AWS for demonstration only.

## Table of Contents
#### [AWS Stack](#aws-stack-1)
#### [Data Methodology and Code](#data-methodology-and-code-1)
#### [Technical Indicators](#technical-indicators-1)
#### [General Model and Mathematical Foundations](#general-model-and-mathematical-foundations-1)
#### [LSTM: Absolute Price](#lstm-absolute-price-1)
#### [LSTM: Percent Change in Price](#lstm-percent-change-in-price-1)
#### [Performance Comparison and Conclusion](#performance-comparison-and-conclusion-1)

## AWS Stack
There are many advantages to building a cloud native ML engine for predictions:

i) First, we do not have to worry about consuming processing power from our local drives

ii) Second, we can integrate a data pipeline to build and load data in a more agile manner to handle multiple production scenarios

iii) Third, it is easy to integrate and set up a notification engine in case we want to automate predictions and output delivery to users

AWS is an excellent and leading cloud platform that allows users to build such end-to-end data architectures. The use case described below is an ideal scenario to leverage in production for a live project.

[**The Ideal Production Data Architecture**](https://aws.amazon.com/blogs/machine-learning/building-machine-learning-workflows-with-aws-data-exchange-and-amazon-sagemaker/)
The blog post contains excellent details regarding an illustrative ML architecture. Additionally, my personal additions to the ideal architecture to make it truly dynamic and user friendly are:

i) Introduce a lambda function to dynamically update the latest data to S3 from AWS Data Exchange subscription each day

ii) Orchestrate a data pipeline in case there are multiple sources of data and transition data between services automatically

iii) Have multiple users poll the latest output through SNS and set up an email notification for results

[<img width="642" alt="AWS Data Architecture" src="https://user-images.githubusercontent.com/36125669/115539086-235a9100-a2cf-11eb-9a90-26062dd071db.png">](https://aws.amazon.com/blogs/machine-learning/building-machine-learning-workflows-with-aws-data-exchange-and-amazon-sagemaker/)

### AWS S3: 
Amazon's Simple Storage Service

S3 is the most fundamental and powerful big data storage solution by AWS in the cloud. The solution can store any structured or unstructured data in several file formats upto a maximum of 5 TB. The current basic set up involves defining a bucket with default encryption to begin our project. 

<details>
<summary>How to set up an S3 Bucket</summary>
  
**Step 1:** 

Create Bucket

<img width="1440" alt="Create S3 Bucket 1" src="https://user-images.githubusercontent.com/36125669/115544714-83eccc80-a2d5-11eb-803b-f3d51546640b.png">

**Step 2:**
Choose region

<img width="1440" alt="Create S3 Bucket 2" src="https://user-images.githubusercontent.com/36125669/115545240-286f0e80-a2d6-11eb-94ef-b08ca461138a.png">

**Step 3:**
Choose encryption type and create bucket

<img width="1440" alt="Create S3 Bucket 3" src="https://user-images.githubusercontent.com/36125669/115545302-3c1a7500-a2d6-11eb-8a95-71809464262e.png">

Our S3 bucket to store input data from our next step is now available!

</details>
  
### AWS Data Exchange
AWS Data Exchange makes it easy to find, subscribe to, and use third-party data in the cloud. Once subscribed to a data product, you can use the AWS Data Exchange API to load data directly into Amazon S3 and then analyze it with a wide variety of AWS analytics and machine learning services. 

For more information watch the [Youtube](https://www.youtube.com/watch?v=2M7S-rsCgfg&t=45s) video from AWS Data Exchange.

<details>
<summary>How to get data from AWS Data Exchange</summary>

**Step 1:** 
For our analysis, the first step was to find relevant S&P 500 data on AWS Data Exchange. The close price data of the S&P 500 was readily available from [FRED: The Federal Reserve Bank of St. Louis.](https://www.stlouisfed.org)

[<img width="1440" alt="Data Exchange S P" src="https://user-images.githubusercontent.com/36125669/115542598-01fba400-a2d3-11eb-8127-f4e469c13314.png">](https://fred.stlouisfed.org)

This dataset contains the S&P 500 Close Price data since April 2011.

**Step 2:**
The next step is to subscribe to the dataset you selected. Usually, the data is available for a free 12 month subscription where you may or may not choose to continue with a paid auto-renewal post that period.

<img width="812" alt="AWS Data Exchange Subscribe" src="https://user-images.githubusercontent.com/36125669/115543255-bd243d00-a2d3-11eb-92e1-ae845da8c057.png">

**Step 3:**
We can now select the S3 bucket we previously created to export the datasets we selected.

<img width="1440" alt="Data Export to S3" src="https://user-images.githubusercontent.com/36125669/115545617-97e4fe00-a2d6-11eb-8710-7a4224d93ee1.png">

</details>

### AWS Sagemaker
Sagemaker is the powerful ML platform from AWS that can be used to leverage several open source and AWS specific ML packages for cutting-edge predictive solutions. There are a host of services and options available as summarized in the diagram with more information to be found on the [AWS Sagemaker Page.](https://aws.amazon.com/sagemaker/)

[<img width="1182" alt="Sagemaker Highlights" src="https://user-images.githubusercontent.com/36125669/115550631-a7674580-a2dc-11eb-8f2f-f5f0bcb8490f.png">](https://aws.amazon.com/sagemaker/)

There are several basic set up steps to using Jupyter Notebooks and Tensorflow in AWS Sagemaker. The path described below is the easiest method from a visual detailing perspective to describe the workflow. Please follow the set up steps for reference.

<details>
<summary>Sagemaker Set Up Steps</summary>

**Step 1:** 
Search for AWS Sagemaker

<img width="1023" alt="Search Sagemaker" src="https://user-images.githubusercontent.com/36125669/115548591-3757c000-a2da-11eb-972e-439c63450514.png">

**Step 2:** 
Select Notebook and Notebook Instance from the left hand menu options

<img width="268" alt="Select Notebook Instance" src="https://user-images.githubusercontent.com/36125669/115548636-476f9f80-a2da-11eb-8f03-49316f63e79f.png">

**Step 3:** 
Select Create Notebook

<img width="1129" alt="Create Notebook" src="https://user-images.githubusercontent.com/36125669/115548691-55252500-a2da-11eb-9da5-6e409bf5be50.png">

**Step 4:** 
Provide a name for your project and select an EC2 instance to deploy your workload: T2 Micro free-tier should be sufficient unless processing GPU instensive ML data

<img width="846" alt="Create Notebook Instance" src="https://user-images.githubusercontent.com/36125669/115548758-6a01b880-a2da-11eb-8e32-a7085e259378.png">

**Step 5:** 
Select IAM: Either create a new IAM execution role or use an exisiting role you may have created previously
<img width="848" alt="Notebook IAM" src="https://user-images.githubusercontent.com/36125669/115548831-80a80f80-a2da-11eb-81b7-7536da2ebee5.png">

**Step 6:** 
Finally Create Notebook
<img width="864" alt="Create Notebook Final" src="https://user-images.githubusercontent.com/36125669/115548907-9a495700-a2da-11eb-90bd-994e2e48e651.png">

**Step 7:** 
Open with Jupyter Lab
<img width="845" alt="Open with Jupyter Lab" src="https://user-images.githubusercontent.com/36125669/115548953-a6351900-a2da-11eb-9085-dd795373a2a7.png">

**Step 7:** 
Select Python Instance with Conda Tensorflow package
<img width="1429" alt="Choose Tensorflow Notebook" src="https://user-images.githubusercontent.com/36125669/115548961-a8977300-a2da-11eb-9b7b-93b03579c1b0.png">

</details>

## Data Methodology and Code
The python notebooks attached layout the steps to import data and make the necessary transformations at each stage till the final results. For brevity, we will only draw on the main themes and highlight important steps to link it with the larger descriptive summary here.

Here are some of the starter codes import relevant libraries and the dataset from S3 (reference the python notebooks for more details).

```
# for sagemaker and iam role
import boto3 # AWS Python SDK
from sagemaker import get_execution_role
role = get_execution_role()

# for tensorflow libraries and modules through sagemaker
import sagemaker.tensorflow 
import tensorflow as tf
from tensorflow import keras

# Import data saved from AWS Data Exchange to S3 bucket
my_bucket = 'stock-price-predictor' # declare bucket name
my_file = 'fred-SP500/dataset/fred-sp500.csv' # declare file path with S3 bucket

# file
data_location = 's3://stock-price-predictor/fred-SP500/dataset/fred-sp500.csv'.format(my_bucket,my_file)
data = pd.read_csv(data_location)
data.head()

```

## Technical Indicators
For a deeper and more intuitive understanding of the data, it is necessary we highlight some of the important technical indicators that are applied in algorithmic trading.

**Moving Average:** An [exponential moving average (EMA)](https://www.investopedia.com/terms/e/ema.asp) is a type of moving average (MA) that places a greater weight and significance on the most recent data points. The exponential moving average is also referred to as the exponentially weighted moving average. An exponentially weighted moving average reacts more significantly to recent price changes than a simple moving average (SMA), which applies an equal weight to all observations in the period.

- **12 Period Moving Average:** Faster moving average characterized by shorter term / retail investors

- **26 Period Moving Average:** Slower moving average that lags based on block trades carried out over time by more institutional investors

**MACD:** [Moving Average Convergence Divergence](https://www.investopedia.com/terms/m/macd.asp) trend-following momentum indicator that shows the relationship between two moving averages of a security’s price. The MACD is calculated by subtracting the 26-period exponential moving average (EMA) from the 12-period EMA.

A more detailed explanation can be found from 
[<img width="1023" alt="AWS Databricks Summit" src="https://user-images.githubusercontent.com/36125669/115554726-5efe5680-a2e1-11eb-9118-e8a107240fde.png">](https://www.youtube.com/watch?v=jlr8QgCxLe4)

**Standard Plot**

S&P 500 Close Price: Apr-2011 to Apr-2021
<img width="1069" alt="S P 500 Close Plot" src="https://user-images.githubusercontent.com/36125669/115553317-dcc16280-a2df-11eb-8f16-73c3dc260039.png">

**Standard Plot with EMA and MACD**

S&P 500 Close Price: Apr-2011 to Apr-2021
<img width="978" alt="EMA MACD 2011" src="https://user-images.githubusercontent.com/36125669/115657577-b516dc80-a369-11eb-8c2f-fe1f6b4759c7.png">

**Theoretical Consideration:** From the plot above we notice that technical indicators over long time horizons tend to be stationary (a concept we will review below) and tend to influence long term buy-sell decisions less from the standpoint of technical indicators. However, when we look at the chart below between 1-Jan-2020 and 15-Apr-2021 we clearly see the variations in the EMAs through MACD. With the red circles showing a negative reversal in trends and a green circle showing a positive upward trend. This also highlights for future consideration the importance of the most recent data when using RNN - weights get increasingly larger the closer we are to the present in a time series - especially while using absolute numbers.

![EMA MACD 2020](https://user-images.githubusercontent.com/36125669/115657581-b5af7300-a369-11eb-8ed6-5fafbe377ed7.jpeg)

## Mathematical Foundations and General Model

#### Mathematical Foundations
While approaching the stock price prediction problem there are several use cases where the absolute values of the close prices of stocks are used for forecasting. However, from a mathematical perspective this creates relative inconsistencies in prediction and the correct way to analyze and forecast such data is to therefore calculate some relative measure of progress that tend toward the following two key properties:

1) **Normalization:** [Normalization](https://en.wikipedia.org/wiki/Normalization_(statistics)) means adjusting values measured on different scales to a notionally common scale, often prior to averaging. In more complicated cases, normalization may refer to more sophisticated adjustments where the intention is to bring the entire probability distributions of adjusted values into alignment. This is especially required when doing a multi-variate analysis with several variables on different time scales and is also useful when analyzing historical data for forecasting on variables that have seen a significant growth over time, i.e, the S&P 500 prices in the last 10 years.

2) **Stationarity:** A stationary time series is one whose statistical properties such as mean, variance, autocorrelation, etc. are all constant over time. Most statistical forecasting methods are based on the assumption that the time series can be rendered approximately stationary (i.e., "stationarized") through the use of mathematical transformations. More detailed description can be found [here](https://people.duke.edu/~rnau/411diff.htm)

The general mathematical foundations for using a single variable, such as the stock close price to forecast future price states through a relative metric like differencing (relative change between periods) comes under the class of models known as [Autoregressive Integrated Moving Averages](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average). 

<img width="1418" alt="ARIMA" src="https://user-images.githubusercontent.com/36125669/115989847-ee15b200-a5f2-11eb-95b2-029f4d03ded2.png">

There are several useful statistical benefits for using ARIMA models. However, for our analysis we will try to explore the benefits of RNN's through LSTM. We will try two versions of our model for review: 
1) Based on **Absolute Prices** and;
2) Based on the **Percentage Log Normal** adjusted prices for normalization and stationarity.

**Transformations**
To achieve stationarity and normalization we will follow some of the guidance of this article by [Quantivity](https://quantivity.wordpress.com/2011/02/21/why-log-returns/):
1) Take the percentage change in price between current and last close.
2) Do a log transformation of the percent stage assumign log normality for normalization

**MODEL PERFORMANCE: Root Mean Square Error**

In statistics, the mean squared error (MSE) of an estimator (of a procedure for estimating an unobserved quantity) measures the average of the squares of the errors — that is, the average squared difference between the estimated values and what is estimated. MSE is a risk function, corresponding to the expected value of the squared error loss. The fact that MSE is almost always strictly positive (and not zero) is because of randomness or because the estimator does not account for information that could produce a more accurate estimate.

RMSE is the square root of the MSE calculated.

**Graphical Plot: Highlighting distance of individual points from the mean**

<img width="791" alt="MSE Graph" src="https://user-images.githubusercontent.com/36125669/115991395-c0346b80-a5fa-11eb-9ad3-cffbeab3a2e3.png">

**Mathematical Formula: RSME**

<img width="500" alt="RMSE Formula" src="https://user-images.githubusercontent.com/36125669/115991532-79934100-a5fb-11eb-926e-4dec0757dc29.png">

We will use this metric to evaluate the outcomes of both our models

#### General Model: LSTM Recurrent Neural Network (RNN)

A recurrent neural network (RNN) is a type of artificial neural network which uses sequential data or time series data. These deep learning algorithms are commonly used for ordinal or temporal problems, such as language translation, natural language processing (NLP), speech recognition, and image captioning; they are incorporated into popular applications such as Siri, voice search, and Google Translate. 

Like feedforward and convolutional neural networks (CNNs), recurrent neural networks utilize training data to learn. They are distinguished by their “memory” as they take information from prior inputs to influence the current input and output. While traditional deep neural networks assume that inputs and outputs are independent of each other, the output of recurrent neural networks depend on the prior elements within the sequence. While future events would also be helpful in determining the output of a given sequence, unidirectional recurrent neural networks cannot account for these events in their predictions.

<img width="512" alt="RNN" src="https://user-images.githubusercontent.com/36125669/115990962-ac880580-a5f8-11eb-9341-3123086e2eec.png">

**The Problem, Short Term Memory**

Recurrent Neural Networks however suffer from short-term memory. If a sequence is long enough, they’ll have a hard time carrying information from earlier time steps to later ones. So if you are trying to process a paragraph of text to do predictions, RNN’s may leave out important information from the beginning.

During back propagation, recurrent neural networks suffer from the vanishing gradient problem. Gradients are values used to update a neural networks weights. The vanishing gradient problem is when the gradient shrinks as it back propagates through time. If a gradient value becomes extremely small, it doesn’t contribute too much learning.

**What is an LSTM?**

This is a popular RNN architecture, which was introduced by Sepp Hochreiter and Juergen Schmidhuber as a solution to vanishing gradient problem. In their [paper](https://www.bioinf.jku.at/publications/older/2604.pdf) (PDF, 388 KB), they work to address the problem of long-term dependencies. That is, if the previous state that is influencing the current prediction is not in the recent past, the RNN model may not be able to accurately predict the current state. 

As an example, let’s say we wanted to predict the italicized words in following, “Alice is allergic to nuts. She can’t eat peanut butter.” 
The context of a nut allergy can help us anticipate that the food that cannot be eaten contains nuts. However, if that context was a few sentences prior, then it would make it difficult, or even impossible, for the RNN to connect the information. 
To remedy this, LSTMs have “cells” in the hidden layers of the neural network, which have three gates–an input gate, an output gate, and a forget gate. These gates control the flow of information which is needed to predict the output in the network.  For example, if gender pronouns, such as “she”, was repeated multiple times in prior sentences, you may exclude that from the cell state.

Check out [Youtube](https://www.youtube.com/watch?v=8HyCNIVRbSU&t=61s) video from ['The A.I. Hacker - Michael Phi'](https://www.youtube.com/channel/UCYpBgT4riB-VpsBBBQkblqQ) for more.

**Model Highlights**
Our model has the following features:
1) Two layer LSTM Model
2) Each layer has 50 neurons
3) Dropout of 0.2 (20%) to prevent overfitting at each LSTM layer
4) Dense layer with 30 neurons followed by single neuron dense output layer
5) **Optimizer:** [Adam](https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c) is an optimization algorithm used instead of the classical [stochastic gradient](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) descent procedure to update network weights iterative based in training data.

```
#Initializing the RNN

model = Sequential()

#Adding the first LSTM layer and some Dropout Regularization
model.add(LSTM(units = 50, return_sequences = True, input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
model.add(Dropout(0.2))
              
#Adding a second LSTM layer and some Dropout Regularization
model.add(LSTM(units = 50, return_sequences = False))
model.add(Dropout(0.2))
              
#Adding the output layer
model.add(Dense(30))
model.add(Dense(1))
```

**Model Time Period**
Both models follow the same time periods:
1) Training Set - Apr 2011 - Dec 2020
2) Test Set - YTD 15-Apr-2021

**Forecast Outputs**
1) Single Step Forecast - Predict Next Close Price (16-Apr-2021)
2) Multi-Step Forecast - Predict Next 6 Days Close Price (16-Apr - 23-Apr -2021)

#### [LSTM: Absolute Price]

**Model Performance Summary**

```
rmse = np.sqrt(np.mean(testPredict - real_stock_price)**2)
rmse

```

#### [LSTM: Percent Change in Price]

**Model Performance Summary**

```
rmse = np.sqrt(np.mean(testPredict - real_stock_price)**2)
rmse

```

#### [Performance Comparison and Conclusion]



## Author
Neil Shastry

## Acknowledgments
I would sincerely like to acknowledge the references and inspirations for this project across a wide range of sources.
1. [AWS Databricks Summit](https://www.youtube.com/watch?v=jlr8QgCxLe4) formed a primary reference for this project. While databricks and AWS DeepAR will be used in future projects - this reference is a useful guide for the general framework of time series and data wrangling in AWS
2. Sepp Hochreiter and Juergen Schmidhuber [paper](https://www.bioinf.jku.at/publications/older/2604.pdf) on the Long-Short Term Memory Problem (LSTM).
3. IBM [blog](https://www.ibm.com/cloud/learn/recurrent-neural-networks) with explanations on RNNs and LSTM
