# Stock Price Prediction with LSTM in AWS Sagemaker on the Cloud

## Background
This is a fun personal project where I wanted to explore some of the math, technical indicators and use deep learning in a cloud native environment to validate this growing hypothesis that machine learning can predict stock prices. For this illustration, I have used the broad market S&P 500 index - however, this project can be readily extended to evaluate other individual or a basket of stocks.

## Objectives
Evaluate time series data of S&P 500 Close price through LSTM RNN and explore prediction feasibility for future close price.

## Tools
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)

## Prerequisite
Sign-up to AWS free-tier through the console to access the services described in this project.

[<img width="1414" alt="AWS Sign-up" src="https://user-images.githubusercontent.com/36125669/115541657-fcea2500-a2d1-11eb-9054-bc411e8b49ba.png">](https://aws.amazon.com/free/)


## Table of Contents
#### [AWS Stack]
#### [Technical Indicators]
#### [Data Wrangling]
#### [Model Build]
#### [Some Mathematical Background]
#### [Prediction Results]

## AWS Stack
There are many advantages to build a cloud native ML engine for predictions:
i) First, we do not have to worry about processing or consuming data from our local drive
ii) Second, we can integrate a data pipeline to build and load data in a more agile manner
iii) Third, it is easy to integrate and set up a notification engine in case we want to automate prediction results and output delivery to users

AWS is an excellent and leading cloud platform that allows users to build such end-to-end data architectures. The use case described below is an ideal scenario to leverage in production for a live project.

[**The Ideal Production Data Architecture**](https://aws.amazon.com/blogs/machine-learning/building-machine-learning-workflows-with-aws-data-exchange-and-amazon-sagemaker/)
The blog post to this image contains excellent details regarding this architecture. Additionally, I would add the following to make it truly dynamic and user friendly:

i) Introduce a lambda function to dynamically update the latest data to S3 from AWS Data Exchange each day

ii) Orchestrate a data pipeline in case there are multiple sources of data and transition data between services automatically

iii) Have multiple users poll the latest output through SNS and set up an email notification for results

[<img width="642" alt="AWS Data Architecture" src="https://user-images.githubusercontent.com/36125669/115539086-235a9100-a2cf-11eb-9a90-26062dd071db.png">](https://aws.amazon.com/blogs/machine-learning/building-machine-learning-workflows-with-aws-data-exchange-and-amazon-sagemaker/)

**AWS S3:** Amazon's Simple Storage Service

S3 is the most fundamental and powerful big data storage solution by AWS in the cloud. The solution can store any structured or unstructured data in several file formats upto a maximum of 5 TB. The set up involves defining a bucket and IAM policy for secure access. 

**Step 1:** 

Create Bucket
<img width="1440" alt="Create S3 Bucket 1" src="https://user-images.githubusercontent.com/36125669/115544714-83eccc80-a2d5-11eb-803b-f3d51546640b.png">

**Step 2:**
Choose region

<img width="1440" alt="Create S3 Bucket 2" src="https://user-images.githubusercontent.com/36125669/115545240-286f0e80-a2d6-11eb-94ef-b08ca461138a.png">

**Step 3:**
Choose encryption type and create bucket

<img width="1440" alt="Create S3 Bucket 3" src="https://user-images.githubusercontent.com/36125669/115545302-3c1a7500-a2d6-11eb-8a95-71809464262e.png">

Our S3 bucket to store data from our next step is now ready!

**AWS Data Exchange**

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

