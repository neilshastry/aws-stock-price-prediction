# Stock Price Prediction with LSTM in AWS Sagemaker on the Cloud

## Background
This is a fun personal project where I wanted to explore some of the math, technical indicators and use deep learning in a cloud native environment to validate this growing hypothesis that machine learning can predict stock prices. For this illustration, I have used the broad market S&P 500 index - however, this project can be readily extended to evaluate other individual or a basket of stocks.

## Objectives
Evaluate time series data of S&P 500 Close price through LSTM RNN and explore prediction feasibility for future close price.

## Tools
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)

## Prerequisite
Sign-up to AWS free-tier through the console:
[<img width="1414" alt="AWS Sign-up" src="https://user-images.githubusercontent.com/36125669/115541134-5a31a680-a2d1-11eb-8d54-00c6b01cf9d4.png">](https://aws.amazon.com/free/?trk=ps_a134p000003yYyZAAU&trkCampaign=acq_paid_search_brand&sc_channel=ps&sc_campaign=acquisition_HK&sc_publisher=google&sc_category=core-main&sc_country=HK&sc_geo=CHNA&sc_outcome=Acquisition&sc_detail=aws&sc_content=Brand_Core_aws_e&sc_matchtype=e&sc_segment=450873684891&sc_medium=ACQ-P%7CPS-GO%7CBrand%7CDesktop%7CSU%7CCore-Main%7CCore%7CHK%7CEN%7CText&s_kwcid=AL!4422!3!450873684891!e!!g!!aws&ef_id=EAIaIQobChMI3NeZy5OP8AIVTVRgCh1XPgm6EAAYASAAEgKdHvD_BwE:G:s&s_kwcid=AL!4422!3!450873684891!e!!g!!aws&all-free-tier.sort-by=item.additionalFields.SortRank&all-free-tier.sort-order=asc&awsf.Free%20Tier%20Types=*all&awsf.Free%20Tier%20Categories=*all)


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

**AWS Data Exchange**
For our analysis, the first step is to find relevant data 

