import figures
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

layout = html.Div(
    children=[
        dbc.Row([
            dbc.Col(children=[
                dcc.Graph(
                    id='jan-apr_values-graph',
                    figure=figures.last_4_months(),
                    style={
                        "margin-top": "15px",
                        "margin-left": "15px",
                        "margin-down": "15px"
                    }
                )
            ], xs=12, sm=12, md=12, lg=6, xl=6),
            dbc.Col(children=[
                dcc.Graph(
                    id='monthly_values-graph',
                    figure=figures.monthly_values_fig(),
                    style={
                        "margin-top": "15px",
                        "margin-right": "15px",
                        "margin-down": "15px"
                    }
                ),
            ], xs=12, sm=12, md=12, lg=6, xl=6)
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id='finance-graph',
                    figure=figures.fin_sa_best_fig("rnn"),
                    style={
                        "margin-top": "15px",
                        "margin-left": "15px",
                        "margin-right": "15px",
                        "margin-down": "15px"
                    }
                ),
            ], xs=12, sm=12, md=12, lg=11, xl=11)
        ]),
        dbc.Row([
            dbc.Col(
                children=[
                    dcc.Graph(
                        id='sa-graph',
                        figure=figures.sa_fig(),
                        style={
                            "margin-top": "40px",
                            "margin-left": "15px",
                            "margin-right": "15px",
                            "margin-down": "15px"
                        }
                    )
                ], xs=12, sm=12, md=12, lg=9, xl=9),
            dbc.Col([
                dbc.Row(
                    children=[
                        dcc.Graph(
                            id='sa-mean-graph',
                            figure=figures.sa_mean_fig(),
                            style={
                                "margin-top": "15px",
                                "margin-left": "15px",
                                "margin-right": "15px",
                            }
                        )
                    ]),
                dbc.Row(
                    children=[
                        dcc.Graph(
                            id='sa-pie_chart-graph',
                            figure=figures.sa_pie_chart_fig(),
                            style={
                                "margin-left": "15px",
                                "margin-right": "15px",
                                "margin-down": "15px"
                            }
                        )
                    ])
            ], xs=12, sm=12, md=12, lg=3, xl=3)
        ])
    ])
