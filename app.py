import dash
from dash_bootstrap_components import Input
from dash_html_components import Output
import figures
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

from apps import home, game_stop, reddit

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )
# server = app.server
# app.config.suppress_callback_exceptions = True

navbar = dbc.NavbarSimple(
    children=[
        html.A("Home", href="#", style={"font-size": 20, "padding": 5, "color": "black"}),
        html.A("About GameStop", href="#gme", style={"font-size": 20, "padding": 5, "color": "black"}),
        html.A("About r/WallStreetBets", href="#reddit", style={"font-size": 20, "padding": 5, "color": "black"})
    ],
    brand="Trade Wisely",
    brand_href="#",
    brand_style={
        "font-size": 25
    }
)
navbar2 = dbc.NavbarSimple(
    fixed='top',
    children=[
        html.A("Home", href="#", style={"font-size": 20, "padding": 5, "color": "black"}),
        html.A("About GameStop", href="#gme", style={"font-size": 20, "padding": 5, "color": "black"}),
        html.A("About r/WallStreetBets", href="#reddit", style={"font-size": 20, "padding": 5, "color": "black"})
    ],
    brand="Trade Wisely",
    brand_href="#",
    brand_style={
        "font-size": 25
    }
)

app.layout = html.Div(
    style={
        "background": "radial-gradient(circle, rgba(2,0,36,1) 0%, rgba(18,74,91,1) 74%, rgba(15,61,82,1) 95%, rgba(19,78,94,1) 100%)"
    },
    children=[
        dbc.Row([
            dbc.Col([
                navbar
            ])
        ]),
        dbc.Row([
            dbc.Col([
                navbar2
            ])
        ]),
        home.layout,
        game_stop.layout,
        dbc.Row(children=[
            dbc.Col([
                html.H2("r/WallStreetBets", id="reddit",
                        style={
                            "margin-top": "15px",
                            "margin-left": "50px",
                            "margin-down": "15px",
                            "margin-right": "15px",
                            "color": "white"
                        }
                        ),
                html.P('''
                            “r/WallStreetBets” is a subreddit, where users post only messages related to the stock market.
                        ''',
                       style={
                           "margin-top": "15px",
                           "margin-left": "50px",
                           "margin-down": "15px",
                           "margin-right": "15px",
                           "color": "white",
                           "font-size": 16
                       }
                       ),
                html.P('''
                            According to  (/r/wallstreetbets metrics (wallstreetbets) n.d.), a website for tracking the grow of
                            subreddits,  “r/WallStreetBets” was founded on January 31, 2012, and has gained since then over 10
                            millions of users, from which more than 8 millions joined the subreddit after the 1st of January, 2021.
                            In average, there are 162,301 active users per week and 155,720 per month. The forum is free of market
                            manipulation, politics and other messages, that are unrelated to stocks, which makes it a good resource
                             for clean data. Moreover, people with little stock exchange experience, or unsure information, are not
                              encouraged share their opinions.
                        ''',
                       style={
                           "margin-top": "15px",
                           "margin-left": "50px",
                           "margin-down": "15px",
                           "margin-right": "15px",
                           "color": "white",
                           "font-size": 16
                       }
                       )
            ], xs=12, sm=12, md=12, lg=6, xl=6),
            dbc.Col(children=[
                html.Br([]),
                html.Img(src=app.get_asset_url('comm_per_day_wallstreetbets.png'), height=400, width=600),
                html.Div("[source: https://subredditstats.com/r/wallstreetbets]", style={"color": "white"})
            ], xs=12, sm=12, md=12, lg=6, xl=6)
        ]),
        html.Br([])
    ])

if __name__ == '__main__':
    app.run_server(debug=True)
