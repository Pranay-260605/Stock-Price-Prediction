import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler


model = load_model('/Users/pranay/Documents/STOCK PRICE PREDICTION/StockPredictionsModel.keras')


st.header('Stock Market Predictor')


stock = st.text_input('Enter Stock Symbol', 'GOOG')


start = '2012-01-01'
end = '2024-12-20'


data = yf.download(stock, start=start, end=end)



data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])


scaler = MinMaxScaler(feature_range=(0, 1))


pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)


x, y = [], []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)


predict = model.predict(x)


scale_factor = 1 / scaler.scale_[0]
predict = predict * scale_factor
y = y * scale_factor


st.subheader('Original Price vs Predicted Price')


result_df = pd.DataFrame({
    'Predicted Price': predict.flatten(),
    'Actual Price': y.flatten()
}, index=data_test.index[100:])


st.line_chart(result_df)
