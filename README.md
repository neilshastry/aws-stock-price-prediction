# Stock Price Prediction with LSTM in AWS Sagemaker on the Cloud

## Background
This is a fun personal project where I wanted to explore some of the math, technical indicators and deep learning in a cloud native environment to validate this growing hypothesis that machine learning can predict stock prices. For this illustration, I have used the broad market S&P 500 index - however, this project can be readily extended to evaluate other individual or a basket of stocks.

## Objectives
Evaluate time series data of S&P 500 Close price through LSTM RNN and explore prediction feasibility for future close price.

## Tools
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)

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

![sagemaker-dataexchange-1.gif](https://aws.amazon.com/blogs/machine-learning/building-machine-learning-workflows-with-aws-data-exchange-and-amazon-sagemaker/)



