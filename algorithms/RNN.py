import math
import os
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def rnn():
    # load the dataset and remove NaN
    dataset = pd.read_csv("D:\licenta\dash-project\datasets\\training\\fin_sa_may_march.csv")
    # dataset = pd.read_csv("D:\licenta\dash-project\datasets\\training\GME_may_march.csv")
    # dataset = pd.read_csv("D:\licenta\dash-project\datasets\GME-monthly.csv")
    dataset.dropna(subset=['Open'], inplace=True)
    dataset.index = pd.to_datetime(dataset['Date'], format='%Y-%m-%d', errors='coerce')

    split_fraction = 0.7  # 70% din date pentru antrenament
    train_split = int(split_fraction * int(dataset.shape[0]))
    learning_rate = 0.001
    batch_size = 128
    epochs = 100
    sequence_length = 1  # o secventa de 1 zile
    step = 1  # analizez fiecare zi

    # split into train and test
    y = dataset['Close']
    x = dataset.drop(['Date', 'High', 'Low', 'Adj Close','Open', 'Close'], axis=1)
    # x = dataset.drop(['Date'], axis=1)

    # normalize data
    def normalize(data):
        data_mean = data.mean()
        data_std = data.std()
        return (data - data_mean) / data_std

    def denormalize(data, std, mean):
        return data * std + mean

    x = normalize(x)
    y_mean = y.mean()
    y_std = y.std()
    y = normalize(y)

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=17)
    x_train, x_test = x[0 : train_split - 1], x[train_split:]
    y_train, y_test = y[0 : train_split - 1], y[train_split:]

    dataset_train = keras.preprocessing.timeseries_dataset_from_array(
        x_train.values,
        y_train.values,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=batch_size,
    )
    dataset_val = keras.preprocessing.timeseries_dataset_from_array(
        x_test.values,
        y_test.values,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=batch_size,
    )

    inputs = keras.layers.Input(shape=(None, 2))
    lstm_out = keras.layers.LSTM(256, activation='tanh')(inputs)
    dense = keras.layers.Dense(128, activation='tanh')(lstm_out)
    outputs = keras.layers.Dense(1, activation='tanh')(dense)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
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

    no_takes = math.ceil(len(x_test) / batch_size)
    predicted_values = []
    for x, y in dataset_val.take(no_takes):
        predicted_list = model.predict(x).tolist()
        for val in predicted_list:
            predicted_values += val

    # Denormalize data
    y_train = denormalize(y_train, y_std, y_mean)
    y_test = denormalize(y_test, y_std, y_mean)
    predicted_values = denormalize(pd.Series(predicted_values), y_std, y_mean)

    history_values = pd.Series(y_train.sort_index().tolist())
    actual_values = pd.Series(y_test.sort_index().tolist())
    predicted_values_df = pd.DataFrame(predicted_values, columns=['Close'])
    predicted_values_df.index = y_test.index
    predicted_values = pd.Series(predicted_values_df['Close'].sort_index().values.tolist())
    y_test_sorted = y_test.sort_index()
    history_date = pd.DataFrame(y_train.index, columns=['Date'])
    date = pd.DataFrame(y_test_sorted.index, columns=['Date'])


    MSE_error = mean_squared_error(actual_values, predicted_values)
    print('Mean Squared Error is {}'.format(MSE_error))

    MAE_error = mean_absolute_error(actual_values, predicted_values)
    print('Mean Absolute Error is {}'.format(MAE_error))

    MAPE_error = mean_absolute_percentage_error(actual_values, predicted_values)
    print('Mean Absolute Percentage Error is {}'.format(MAPE_error))

    with open('D:\\licenta\\dash-project\\datasets\\rnn_sa_history.csv', 'w+') as fd:
        fd.write("date,value\n")
        for x in range(len(history_values)):
            fd.write(str(history_date.loc[x, 'Date']) + "," + str(history_values[x]) + "\n")
        fd.close()
    with open('D:\\licenta\\dash-project\\datasets\\rnn_sa_predicted.csv', 'w+') as fd:
        fd.write("date,value\n")
        for x in range(len(actual_values)):
            fd.write(str(date.loc[x, 'Date']) + "," + str(predicted_values[x]) + "\n")
        fd.close()
    with open('D:\\licenta\\dash-project\\datasets\\rnn_sa_real.csv', 'w+') as fd:
        fd.write("date,value\n")
        for x in range(len(actual_values)):
            fd.write(str(date.loc[x, 'Date']) + "," + str(actual_values[x]) + "\n")
        fd.close()

    try:
        os.remove("D:\licenta\dash-project\plots\\rnn.png")
    except:
        print('no previous plot')
    plt.clf()
    plt.plot(history_date.loc[::, 'Date'], history_values, color = 'black', label = 'History')
    plt.plot(date.loc[::, 'Date'], actual_values, color = 'red', label = 'Real Price')
    plt.plot(date.loc[::, 'Date'], predicted_values, color = 'blue', label = 'Predicted Price')
    # plt.title('Gold Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig('D:\licenta\dash-project\plots\\rnn.png')
    return MAPE_error

rnn()