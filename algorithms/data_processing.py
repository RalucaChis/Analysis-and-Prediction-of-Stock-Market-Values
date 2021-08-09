import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

def preprocess_data(dataset_file):
    dataset = pd.read_csv(dataset_file)
    # dataset['Date'] = pd.to_datetime(dataset['Date'])
    # dataset.set_index('Date', inplace=True)
    dataset.dropna(subset=['Open'], inplace=True)
    return dataset

def plot_dataset(dataset):
    plt.plot(dataset.loc[::, 'Date'], dataset[::].get('Close'), color='red')
    plt.xticks(np.arange(0, 608, step=110))
    plt.title('Gold Stock Price In The Last 10 Years')
    plt.xlabel('Time')
    plt.ylabel('Gold Open Price')
    plt.legend()
    plt.show()

# Shows he correlation between different features.
def heatmap(data):
    plt.matshow(data.corr())
    plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.show()

# graphs to show seasonal_decompose
def seasonal_decomposing (dataset):
    df_close = dataset['Close']
    df_close.plot()
    pyplot.show()
    result = seasonal_decompose(df_close, model='multiplicative', period=30)
    # return result._trend
    fig = result.plot()
    fig.set_size_inches(16, 9)
    fig.show()
    # plt.plot(result.trend)
    # plt.show()

# dataset = pd.read_csv("D:\licenta\dash-project\datasets\\training\\fin_sa_may_march.csv")
# dataset.dropna(subset=['Open'], inplace=True)
# seasonal_decomposing(dataset)
# dataset['Date'] = pd.to_datetime(dataset['Date'])
# dataset.set_index('Date', inplace=True)
# scatter_data(dataset)
# y = dataset['Open']
# y.index = pd.to_datetime(dataset['Date'], format='%Y-%m-%d', errors='coerce')
# y = y.asfreq('next_monday')
# seasonal_decompose(y)

def trend_plot (ts):
    plt.plot(ts)
    plt.ylabel('Closing price')
    plt.grid()
    plt.tight_layout()
    plt.savefig('plots/trend.png')
    plt.show()
# ts = dataset['Close']
# trend_plot(ts)

def test_stationarity(timeseries):
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value

    critical_value = dftest[4]['5%']
    test_statistic = dftest[0]
    alpha = 1e-3
    pvalue = dftest[1]
    if pvalue < alpha and test_statistic < critical_value:  # null hypothesis: x is non stationary
        print("X is stationary")
        return True
    else:
        print("X is not stationary")
        return False

# ts = dataset['Close']
# ts_diff = pd.Series(ts)
# print(test_stationarity(ts))