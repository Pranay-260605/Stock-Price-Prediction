# Stock Price Prediction

This repository contains a machine learning model for predicting stock prices based on historical data. The model uses a Long Short-Term Memory (LSTM) neural network to forecast stock prices, which is implemented using TensorFlow and Keras.

This project also includes a Streamlit-based web application where users can input a stock symbol, and the model will predict the stock's future prices. The predictions are plotted against the actual prices for comparison.

## Table of Contents
- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Model Description](#model-description)



## Overview

This project provides an end-to-end solution for stock price prediction. It uses historical stock price data to train a machine learning model and deploys a user-friendly web application where users can see the predicted stock prices in comparison to the actual prices.

The web application is built with Streamlit and uses data from Yahoo Finance API (`yfinance`) to download the stock data. A pre-trained LSTM model is used to make predictions, which are then visualized on a line chart in the app.

## Technologies Used
- **Python**: The primary programming language used for the project.
- **TensorFlow & Keras**: For building and training the LSTM model.
- **Streamlit**: A fast way to build and deploy web applications.
- **pandas & numpy**: For data manipulation and numerical operations.
- **yfinance**: For fetching historical stock data from Yahoo Finance.
- **scikit-learn**: For data preprocessing (e.g., MinMax scaling).
- **matplotlib**: For visualizations (although Streamlit's line chart is used in the app).

## Model Description

The model is built using Long Short-Term Memory (LSTM), a type of recurrent neural network (RNN) that is well-suited for time-series prediction tasks like stock price forecasting. Here's a brief overview of how the model is structured:

1. **Data Preprocessing**: The data is first scaled using MinMaxScaler to ensure that all features are between 0 and 1.
2. **Training Data**: The data is divided into training and test datasets (80% for training, 20% for testing).
3. **Model Architecture**: The LSTM network consists of multiple LSTM layers with Dropout layers to prevent overfitting. The final output layer is a dense layer with one neuron, which outputs the predicted stock price.
4. **Training**: The model is trained for 50 epochs using the Adam optimizer and Mean Squared Error (MSE) as the loss function.
5. **Prediction**: The model is used to predict future stock prices based on the last 100 days of historical data.





