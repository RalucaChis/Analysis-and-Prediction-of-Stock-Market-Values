import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def volume_analysis():
    dataset = pd.read_csv("D:\licenta\django-backend\charts_service\datasets\GC=F-daily.csv")
    dataset.dropna(subset=['Open'], inplace=True)

    plt.clf()
    plt.bar(dataset.loc[::, 'Date'], dataset[::]['Volume'], color='red', label='Volume', width=5)
    plt.legend()
    plt.xticks(np.arange(0, len(dataset), 110), dataset.Date[0:len(dataset):110])
    plt.twinx()
    plt.plot(dataset.loc[::, 'Date'], dataset[::]['Close'], color='black', label='Price')
    plt.title('Volume Analysis')
    plt.xlabel('Date')
    plt.legend()
    plt.xticks(np.arange(0, len(dataset), 110), dataset.Date[0:len(dataset):110])
    plt.savefig('D:\\licenta\\django-backend\\charts_service\\plots\\volume.png')

def sa_volume_analysis():
    dataset = pd.read_csv("D:\licenta\dash-project\datasets\sa_vol_may_march.csv")

    plt.clf()
    plt.bar(dataset.loc[::, 'Date'], dataset[::]['Volume'], color='red', label='Volume', width=5)
    plt.legend()
    plt.xticks(np.arange(0, len(dataset), 110), dataset.Date[0:len(dataset):110])
    plt.twinx()
    plt.plot(dataset.loc[::, 'Date'], dataset[::]['Sentiment'], color='black', label='Sentiment')
    plt.title('Volume Analysis')
    plt.xlabel('Date')
    plt.legend()
    plt.xticks(np.arange(0, len(dataset), 110), dataset.Date[0:len(dataset):110])
    plt.savefig('D:\licenta\dash-project\plots\\volume.png')

# sa_volume_analysis()