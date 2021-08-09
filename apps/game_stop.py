import figures
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

layout = html.Div(
    children=[
        html.Br([]),
        dbc.Row(children=[
            dbc.Col([
                html.H2("What led to the huge unpredictable success of GameStop?", id="gme",
                        style={
                            "margin-top": "15px",
                            "margin-left": "50px",
                            "margin-down": "15px",
                            "margin-right": "15px",
                            "color": "white",
                        }
                        ),
                html.P('''
                    Well, according to some publications, the company was declining,
                    accentuated by the pandemic. The big investors on WallStreet bet that in the next period the prices
                     will continue to fall and this seemed a certain fact. However, users of the "r / WallStreetBets"
                     (r/wallstreetbets 2021) subreddit felt that the company's value was undervalued and began to buy
                     options to raise prices and succeeded.
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
                    Short sellers had to buy shares to limit their losses, so the
                     price increased. The peak was reached on January 27, 2021, with a closing price
                      of 354,000 USD, compared to 3,840 USD a year ago.
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
                    [sources: https: // edition.cnn.com / 2021 / 01 / 27 / investing / gamestop - reddit - stock / index.html; https://www.cnbc.com/2021/01/27/gamestop-mania-explained-how-the-reddit-retail-trading-crowd-ran-over-wall-street-pros.html]
                ''',

                       style={
                           "margin-top": "15px",
                           "margin-left": "50px",
                           "margin-down": "15px",
                           "margin-right": "15px",
                           "color": "white",
                           "font-size": 10
                       }
                       )
            ], xs=12, sm=12, md=12, lg=4, xl=4),
            dbc.Col(children=[
                html.Br([]),
                dcc.Graph(
                    id="3-finance-graph",
                    figure=figures.fin_3_sa_best_fig(),
                    style={
                        "margin-top": "15px",
                        "margin-left": "15px",
                        "margin-down": "15px"
                    }
                )
            ], xs=12, sm=12, md=12, lg=8, xl=8)
        ])
    ])
