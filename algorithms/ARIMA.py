import os
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import re
import numpy as np
from sklearn.decomposition import PCA
from algorithms import data_processing


def arima():
    split_fraction = 0.7
    dataset = data_processing.preprocess_data("D:\licenta\dash-project\datasets\\training\\fin_sa_may_march.csv")
    train_data, test_data = dataset[0:int(len(dataset) * split_fraction)], dataset[int(len(dataset) * split_fraction):]
    train_data = train_data['Close'].values
    test_data = test_data['Close'].values

    # data_processing.seasonal_decomposing(dataset)
    model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                                 test='adf',  # use adftest to find optimal 'd'
                                 max_p=5, max_q=5,  # maximum p and q
                                 m=1,  # frequency of series
                                 d=None,  # let model determine 'd'
                                 seasonal=False,  # No Seasonality
                                 start_P=0,
                                 D=0,
                                 trace=True,
                                 error_action='ignore',
                                 suppress_warnings=True,
                                 stepwise=True)

    summary_string = str(model_autoARIMA.summary())
    param = re.findall('SARIMAX\(([0-9]+), ([0-9]+), ([0-9]+)', summary_string)
    p, d, q = int(param[0][0]), int(param[0][1]), int(param[0][2])

    # model_autoARIMA.plot_diagnostics(figsize=(15,8))
    # plt.show()
    history = [x for x in train_data]
    model_predictions = []

    # VAR 1
    # for time_point in range(len(test_data)):
    #     model = ARIMA(history, order=(p, d, q))
    #     model_fit = model.fit(disp=0, transparams=False)
    #     output = model_fit.forecast()
    #     yhat = output[0]
    #     model_predictions.append(yhat)
    #     true_test_value = test_data[time_point]
    #     history.append(true_test_value)

    # VAR 2
    history = [x for x in train_data]
    model_predictions = []
    model = ARIMA(history, order=(p,d,q))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast(len(test_data))

    for time_point in range(len(test_data)):
        yhat = output[0][time_point]
        model_predictions.append(yhat)
        # true_test_value = test_data[time_point]
        # history.append(true_test_value)

    MSE_error = mean_squared_error(test_data, model_predictions)
    print('Mean Squared Error is ' + str(MSE_error))

    MAE_error = mean_absolute_error(test_data, model_predictions)
    print('Mean Absolute Error is ' + str(MAE_error))

    MAPE_error = mean_absolute_percentage_error(test_data, model_predictions)
    print('Mean Absolute Percentage Error is ' + str(MAPE_error))

    dataset.dropna(subset=['Open'], inplace=True)
    history_date = pd.DataFrame(dataset[:int(len(dataset)*split_fraction)]['Date'], columns=['Date'])
    date = pd.DataFrame(dataset[int(len(dataset)*split_fraction):]['Date'], columns=['Date'])
    date = [x for x in date['Date']]
    model_predictions = [x[0] for x in model_predictions]

    with open('D:\\licenta\\dash-project\\datasets\\arima_history.csv', 'w+') as fd:
        fd.write("date,value\n")
        for x in range(len(train_data)):
            fd.write(str(history_date.loc[x, 'Date']) + "," + str(train_data[x]) + "\n")
        fd.close()
    with open('D:\\licenta\\dash-project\\datasets\\arima_predicted.csv', 'w+') as fd:
        fd.write("date,value\n")
        for x in range(len(model_predictions)):
            fd.write(str(date[x]) + "," + str(model_predictions[x]) + "\n")
        fd.close()
    with open('D:\\licenta\\dash-project\\datasets\\arima_real.csv', 'w+') as fd:
        fd.write("date,value\n")
        for x in range(len(test_data)):
            fd.write(str(date[x]) + "," + str(test_data[x]) + "\n")
        fd.close()

    # try:
    #     os.remove("charts_service/plots/arima.png")
    # except:
    #     print('no previous plot')

    # plt.clf()
    # plt.plot(dataset[:int(len(dataset)*0.7)].index, train_data, color = 'black', label = 'History')
    # plt.plot(dataset[int(len(dataset)*0.7):].index, test_data, color='red', label='Real Gold Stock Price')
    # plt.plot(dataset[int(len(dataset)*0.7):].index, model_predictions, color='blue',label='Predicted Gold Stock Price')
    # plt.title('Gold Stock Price Prediction')
    # plt.xlabel('Date')
    # plt.ylabel('Gold Stock Price')
    # plt.xticks(np.arange(0,len(dataset),100), dataset.Date[0:len(dataset):100])
    # plt.legend()
    # plt.savefig('D:\licenta\dash-project\plots\\arima.png')
    return MAPE_error

arima()