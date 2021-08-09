from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import re
from tensorflow import keras
import math
from algorithms import data_processing

def hybrid():
    split_fraction= 0.7
    dataset = data_processing.preprocess_data("D:\licenta\dash-project\datasets\\training\GME_may_march.csv")
    dataset.index = pd.to_datetime(dataset['Date'], format='%Y-%m-%d', errors='coerce')
    train_data, test_data = dataset[0:int(len(dataset) * 0.7)], dataset[int(len(dataset) * 0.7):]
    train_data = train_data['Close'].values
    test_data = test_data['Close'].values

    # train_data = trend + seasonality + white noise
    result = seasonal_decompose(train_data, model='multiplicative', freq=30, extrapolate_trend='freq')
    train_trend = result.trend
    train_seasonality = result.seasonal * result.resid
    train_white_noise = result.resid

    # test_data = trend + seasonality + white noise
    result = seasonal_decompose(test_data, model='multiplicative', freq=30, extrapolate_trend='freq')
    test_trend = result.trend
    test_seasonality = result.seasonal * result.resid
    test_white_noise = result.resid

    # apply arima for train_trend
    model_autoARIMA = auto_arima(train_trend, trace=True, error_action='ignore', suppress_warnings=True)

    summary_string = str(model_autoARIMA.summary())
    param = re.findall('SARIMAX\(([0-9]+), ([0-9]+), ([0-9]+)', summary_string)
    p, d, q = int(param[0][0]), int(param[0][1]), int(param[0][2])

    history = [x for x in train_trend]
    arima_predictions = []
    model = ARIMA(history, order=(p, d, q))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast(len(test_trend))

    for time_point in range(len(test_trend)):
        yhat = output[0][time_point]
        arima_predictions.append(yhat)
        # true_test_value = test_data[time_point]
        # history.append(true_test_value)

    MSE_error = mean_squared_error(test_trend, arima_predictions)
    print('Mean Squared Error for ARIMA {}'.format(MSE_error))

    MAPE_error = mean_absolute_percentage_error(test_trend, arima_predictions)
    print('Mean Absolute Percentage Error is {}'.format(MAPE_error))

    # apply lstm for seasonality + white noise
    learning_rate = 0.001
    batch_size = 128
    epochs = 100
    sequence_length = 1  # o secventa de 1 zile
    step = 1  # analizez fiecare zi

    pd_train_seasonality = pd.DataFrame(train_seasonality, index=dataset[:int(len(dataset) * 0.7)].index,
                                        columns=['Close'])
    pd_test_seasonality = pd.DataFrame(test_seasonality, index=dataset[int(len(dataset) * 0.7):].index,
                                       columns=['Close'])

    dataset_train = keras.preprocessing.timeseries_dataset_from_array(
        pd_train_seasonality,
        pd_train_seasonality,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=batch_size,
    )
    dataset_val = keras.preprocessing.timeseries_dataset_from_array(
        pd_test_seasonality,
        pd_test_seasonality,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=batch_size,
    )
    inputs = keras.layers.Input(shape=(None, 1))
    lstm_out = keras.layers.LSTM(256)(inputs)
    dense = keras.layers.Dense(128)(lstm_out)
    outputs = keras.layers.Dense(1)(dense)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    # optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    model.summary()

    path_checkpoint = "model_checkpoint.h5"
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)
    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )
    history = model.fit(
        dataset_train,
        epochs=epochs,
        validation_data=dataset_val,
        callbacks=[es_callback, modelckpt_callback],
    )
    model.load_weights(path_checkpoint)

    no_takes = math.ceil(len(test_seasonality) / batch_size)
    lstm_predictions = []
    for x, y in dataset_val.take(no_takes):
        predicted_list = model.predict(x).tolist()
        for val in predicted_list:
            lstm_predictions += val

    # build predicted data
    predicted_data = []
    for x in range(len(lstm_predictions)):
        val = lstm_predictions[x] * arima_predictions[x]
        predicted_data.append(val)

    MSE_error = mean_squared_error(test_data, predicted_data)
    print('Mean Squared Error is {}'.format(MSE_error))

    MAE_error = mean_absolute_error(test_data, predicted_data)
    print('Mean Absolute Error is {}'.format(MAE_error))

    MAPE_error = mean_absolute_percentage_error(test_data, predicted_data)
    print('Mean Absolute Percentage Error is {}'.format(MAPE_error))

    history_date = pd.DataFrame(dataset[:int(len(dataset) * split_fraction)]['Date'], columns=['Date'])
    date = pd.DataFrame(dataset[int(len(dataset) * split_fraction):]['Date'], columns=['Date'])
    date = [x for x in date['Date']]
    history_date = [x for x in history_date['Date']]

    with open('D:\\licenta\\dash-project\\datasets\\hybrid_history.csv', 'w+') as fd:
        fd.write("date,value\n")
        for x in range(len(train_data)):
            fd.write(str(history_date[x]) + "," + str(train_data[x]) + "\n")
        fd.close()
    with open('D:\\licenta\\dash-project\\datasets\\hybrid_predicted.csv', 'w+') as fd:
        fd.write("date,value\n")
        for x in range(len(predicted_data)):
            fd.write(str(date[x]) + "," + str(predicted_data[x]) + "\n")
        fd.close()
    with open('D:\\licenta\\dash-project\\datasets\\hybrid_real.csv', 'w+') as fd:
        fd.write("date,value\n")
        for x in range(len(test_data)):
            fd.write(str(date[x]) + "," + str(test_data[x]) + "\n")
        fd.close()

    # if os.path.exists("charts_service/plots/hybrid.png"):
    #     os.remove("charts_service/plots/hybrid.png")
    # else:
    #     print("The file does not exist")
    #
    plt.clf()
    plt.plot(dataset[:int(len(dataset) * split_fraction)].index, train_data, color='black', label='History')
    plt.plot(dataset[int(len(dataset) * split_fraction):].index, test_data, color='red', label='Real Price')
    plt.plot(dataset[int(len(dataset) * split_fraction):].index, predicted_data, color='blue', label='Predicted Price')
    # plt.title('Gold Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig('D:\licenta\dash-project\plots\hybrid.png')
    return MAPE_error

hybrid()