import figures
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from app import app

layout=html.Div(
    children=[
        dbc.Row(children=[
            dbc.Col([
                html.H2("r/WallStreetBets",
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
                      encouraged to share their opinions. The graph on the right represents the daily volume of messages on 
                      this subreddit.
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
                html.Div("[source: https://subredditstats.com/r/wallstreetbets]",style={"color":"white"})
            ], xs=12, sm=12, md=12, lg=6, xl=6)
        ]),
        html.Br([]),
        html.Br([]),
        html.Br([])
    ])