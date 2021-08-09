import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
from plotly.figure_factory import create_quiver
import math
import numpy as np

text_color = "white"
bg_color = "#134E5E"

def sa_fig():
    dataset_sa_vol = pd.read_csv("D:\licenta\dash-project\datasets\sa_vol_may_march.csv")

    fig = go.Figure(
        data=[go.Scatter(x=dataset_sa_vol['Date'][:], y=dataset_sa_vol['Sentiment'][:], name="Positivity Rate")])
    # fig = go.Figure(data=[go.Scatter(x=dataset_sa_vol['Date'][-60:], y=dataset_sa_vol['Sentiment'][-60:], name="Positivity Rate")])
    fig.update_yaxes(title_text="Positivity Rate")
    fig.update_layout(
        title_text="Sentimental Value of Reddit Messages",
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font_color=text_color,
        height=450,
        # margin=dict(l=70, r=20, t=50, b=50),
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def sa_mean_fig():
    dataset_sa_vol = pd.read_csv("D:\licenta\dash-project\datasets\sa_vol_may_march.csv")
    sum = 0
    prev_sum = 0
    for x in dataset_sa_vol['Sentiment'][-10:]:
        sum += x
    sum /= 10
    for x in dataset_sa_vol['Sentiment'][-20:-10]:
        prev_sum += x
    prev_sum /= 10

    color = 'rgb(239,85,59)'
    if sum >= 0.5:
        color = 'rgb(0,204,150)'
    fig = go.Figure(go.Indicator(
        domain={'x': [0, 1], 'y': [0, 1]},
        value=sum,
        mode="gauge+number+delta",
        title={'text': "Positivity Rate for the Last 10 Days", 'font': {'size': 14}},
        delta={'reference': prev_sum},
        gauge={'axis': {'range': [0, 1]},
               'steps': [
                   {'range': [0, 1], 'color': "white"}],
               'bar': {'color': color}
               }))
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=0),
        width=280,
        height=250,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font_color=text_color
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def sa_pie_chart_fig():
    dataset_sa_vol = pd.read_csv("D:\licenta\dash-project\datasets\sa_vol_may_march.csv")
    # dataset_sa_vol = dataset_sa_vol[-10:]
    sa_dict = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    for x in dataset_sa_vol['Sentiment']:
        if x > 0.5:
            sa_dict['Positive'] += 1
        elif x < 0.5:
            sa_dict['Negative'] += 1
        else:
            sa_dict['Neutral'] += 1

    df = pd.DataFrame({
        "Sentiment": ["Neutral", "Negative", "Positive"],
        "Volume": [sa_dict['Neutral'], sa_dict['Negative'], sa_dict['Positive']],
    })
    fig = go.Figure(data=[go.Pie(labels=df['Sentiment'], values=df['Volume'], textinfo='label+percent',
                                 insidetextorientation='radial',
                                 # title={'text': "Reddit Messages", 'font': {'size': 14}},
                                 )])
    fig.update_layout(
        margin=dict(l=20, r=20, t=0, b=10),
        width=280,
        height=250,
        # title_text="Sentimental value of Reddit Messages",
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font_color=text_color,
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def vol_trans_msg_fig():
    dataset_sa_vol = pd.read_csv("D:\licenta\dash-project\datasets\sa_vol_may_march.csv")
    dataset_vol_trans = pd.read_csv("D:\licenta\dash-project\datasets\\training\GME_may_march.csv")

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=dataset_sa_vol['Date'], y=dataset_sa_vol['Volume'], name="Volume of Daily Messages"),
        secondary_y=True,
    )
    fig.add_trace(
        go.Bar(x=dataset_vol_trans['Date'], y=dataset_vol_trans['Volume'], name="Volume of Transactions"),
        secondary_y=False,
    )
    # Add figure title
    # fig.update_layout(
    #     title_text="Double Y Axis Example"
    # )
    # Set x-axis title
    # fig.update_xaxes(title_text="xaxis title")

    fig.update_yaxes(title_text="Volume of Daily Messages", secondary_y=True)
    fig.update_yaxes(title_text="Volume of Transactions", secondary_y=False)
    return fig


def fin_sa_best_fig(dataset):
    dataset_history = pd.read_csv("D:\\licenta\\dash-project\\datasets\\" + dataset + "_history.csv")
    dataset_predicted = pd.read_csv("D:\\licenta\\dash-project\\datasets\\" + dataset + "_predicted.csv")
    dataset_real = pd.read_csv("D:\\licenta\\dash-project\\datasets\\" + dataset + "_real.csv")

    fig = go.Figure(data=[go.Scatter(x=dataset_history['date'], y=dataset_history['value'], name="Historical Price")])
    fig.add_scatter(x=dataset_predicted['date'], y=dataset_predicted['value'], name="Predicted Price")
    fig.add_scatter(x=dataset_real['date'], y=dataset_real['value'], name="Real Price")
    fig.update_layout(
        title_text="Price Prediction",
        font=dict(
            size=12,
        ),
        height=350,
        width=1230,
        # margin=dict(l=50, r=50, t=40, b=30),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font_color=text_color
    )
    fig.update_xaxes(showgrid=False, gridwidth=0.5, gridcolor='lightgray')
    fig.update_yaxes(showgrid=False, gridwidth=0.5, gridcolor='lightgray')
    return fig

def fin_3_sa_best_fig():
    dataset_history = pd.read_csv("D:\\licenta\\dash-project\\datasets\\rnn_history.csv")
    dataset_rnn_predicted = pd.read_csv("D:\\licenta\\dash-project\\datasets\\rnn_predicted.csv")
    dataset_rnn_sa_predicted = pd.read_csv("D:\\licenta\\dash-project\\datasets\\rnn_sa_predicted.csv")
    dataset_hybrid_predicted = pd.read_csv("D:\\licenta\\dash-project\\datasets\\hybrid_predicted.csv")
    dataset_real = pd.read_csv("D:\\licenta\\dash-project\\datasets\\rnn_real.csv")

    fig = go.Figure(data=[go.Scatter(x=dataset_history['date'], y=dataset_history['value'], name="Historical Price")])
    fig.add_scatter(x=dataset_rnn_predicted['date'], y=dataset_rnn_predicted['value'], name="RNN Predicted Price",visible='legendonly')
    fig.add_scatter(x=dataset_rnn_sa_predicted['date'], y=dataset_rnn_sa_predicted['value'], name="RNN with SA Predicted Price",visible='legendonly')
    fig.add_scatter(x=dataset_hybrid_predicted['date'], y=dataset_hybrid_predicted['value'], name="Hybrid Predicted Price",visible='legendonly')
    fig.add_scatter(x=dataset_real['date'], y=dataset_real['value'], name="Real Price",visible='legendonly')
    fig.update_layout(
        title_text="Price Prediction",
        font=dict(
            size=12,
        ),
        height=400,
        width=800,
        margin=dict(l=50, r=50, t=40, b=30),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font_color=text_color
    )
    fig.update_xaxes(showgrid=False, gridwidth=0.5, gridcolor='lightgray')
    fig.update_yaxes(showgrid=False, gridwidth=0.5, gridcolor='lightgray')
    return fig

def monthly_values_fig():
    dataset_fin_vol = pd.read_csv("D:\licenta\dash-project\datasets\GME-monthly.csv")
    # dataset_fin_vol = pd.read_csv("D:\licenta\dash-project\datasets\GME-iun20-21.csv")
    ema = dataset_fin_vol['Close'].ewm(span=40).mean()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=dataset_fin_vol['Date'], y=dataset_fin_vol['Close'], name="Stock Closing Price"),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(x=dataset_fin_vol['Date'], y=ema, name="Trend"),
        secondary_y=True,
    )
    fig.add_trace(
        go.Bar(x=dataset_fin_vol['Date'], y=dataset_fin_vol['Volume'], name="Volume of Transactions"),
        secondary_y=False,
    )
    fig.update_layout(
        title_text="Price Evolution in the Last 2 Decades",
        font=dict(
            size=12,
        ),
        height=250,
        width=600,
        margin=dict(l=20, r=50, t=50, b=30),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font_color=text_color
    )
    fig.update_yaxes(showgrid=False, secondary_y=False)
    # fig.update_yaxes(title_text="Stock Closing Price", secondary_y=True)
    # fig.update_yaxes(title_text="Volume of Transactions", secondary_y=False)
    return fig


def last_4_months():
    df = pd.read_csv("D:\licenta\dash-project\datasets\GME_jan_apr_2021.csv")
    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                                         open=df['Open'],
                                         high=df['High'],
                                         low=df['Low'],
                                         close=df['Close'])])
    fig.update_layout(
        title_text="Price Evolution in the Last 4 Months",
        font=dict(
            size=12,
        ),
        height=250,
        width=600,
        margin=dict(l=50, r=20, t=50, b=30),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font_color=text_color,
    )
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray')
    return fig
