from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import pathlib
import os
from app import app

import numpy as np
import pickle as plk
import pwlf
from apps._fun import seperate_data_by_Schedule, Energy_bar_graph

layout = dbc.Container([
    # ---------- Hidden Values
    html.Div(id='data-est', style={'display': 'none'}),
    html.Div(id='page_load_trigger_est', style={'display': 'none'}),

    html.Div(id='steam-data-est', style={'display': 'none'}),
    html.Div(id='water-data-est', style={'display': 'none'}),
    html.Div(id='elec-data-est', style={'display': 'none'}),
    # -----------------

    dbc.Row(
        html.H1('Energy Estimator')
    ),

    dbc.Row([
        dbc.Col([
            html.H4(id='building_title', children=[]),
            dcc.Checklist(id='adjust-individually-checkbox',
                          options=[{'label': ' Adjust Schedule Individually', 'value': True}]),
        ]),
        dbc.Col([
            html.H4(id='year_title', children=[])
        ]),
        dbc.Col([])

    ]),
    dbc.Row([
        dbc.Col([
            html.H6("Steam"),
            dcc.Graph(id='steam-energy-graph', figure={}),
            html.Br(),
            html.P("Proposed Schedule:", id='steam-slider-text'),
            dcc.Checklist(id='steam-weekend-checkbox',
                          options=[{'label': ' Include weekend', 'value': False}]),
            dcc.RangeSlider(id='steam-energy-slider',
                            min=0, max=24, step=1, value=[5, 17], pushable=2,
                            tooltip={"placement": "bottom", "always_visible": True}
                            ),
            html.Br()

        ]),
        dbc.Col([
            html.H6("Chilled Water"),
            dcc.Graph(id='water-energy-graph', figure={}),
            html.Br(),
            html.P("Proposed Schedule", id='water-slider-text'),
            dcc.Checklist(id='water-weekend-checkbox',
                          options=[{'label': ' Include weekend', 'value': False}]),
            dcc.RangeSlider(id='water-energy-slider',
                            min=0, max=24, step=1, value=[5, 17], pushable=2,
                            tooltip={"placement": "bottom", "always_visible": True}
                            ),
            html.Br()
        ]),

        dbc.Col([
            html.H6("Electricity"),
            dcc.Graph(id='elec-energy-graph', figure={}),
            html.Br(),
            html.P("Proposed Schedule", id='elec-slider-text'),
            dcc.Checklist(id='elec-weekend-checkbox',
                          options=[{'label': ' Include weekend', 'value': False}]),
            dcc.RangeSlider(id='elec-energy-slider',
                            min=0, max=24, step=1, value=[5, 17], pushable=2,
                            tooltip={"placement": "bottom", "always_visible": True}
                            ),
            html.Br()

        ])
    ]),
    dbc.Row([
        dcc.Checklist(id='main-weekend-checkbox', value=[],
                      options=[{'label': ' Include weekend', 'value': False}]),
        dcc.RangeSlider(id='main-energy-slider',
                        min=0, max=24, step=1, value=[5, 17], pushable=2,
                        tooltip={"placement": "bottom", "always_visible": True}
                        ),
        html.P(id='summary-text')
    ])

], fluid=True)


@app.callback(
    [Output(component_id='data-est', component_property='children'),
     Output(component_id='building_title', component_property='children'),
     Output(component_id='year_title', component_property='children')],
    [Input(component_id='page_load_trigger_est', component_property='children')]
)
def page_load_trigger(t):
    # Get model info
    PATH = pathlib.Path(__file__).parent
    DATA_PATH = PATH.joinpath("../saved").resolve()

    with open(DATA_PATH.joinpath(f'model.pkl'), 'rb') as f:
        model = plk.load(f)
    building, year, steam_sch, _, _, _, water_sch, _, _, _, elec_sch, _, _, _ = model

    building_header = ['Building: ', building]
    year_header = ['Year: ', str(year)]

    # Get Weather Data
    DATA_PATH = PATH.joinpath("../data").resolve()
    with open(DATA_PATH.joinpath(f'Supporting/Weather/Ottawa_TMY.pkl'), 'rb') as f:
        df = plk.load(f)

    df = df['temperature']

    # Convert to Json for storage
    data = df.to_json(date_format='iso', orient='split')

    return data, building_header, year_header  # , steam_sch, water_sch, elec_sch


# ---------------- Graphs
@app.callback(
    [Output(component_id='steam-energy-graph', component_property='figure'),
     Output(component_id='steam-data-est', component_property='children')],
    [Input(component_id='data-est', component_property='children'),
     Input(component_id='steam-energy-slider', component_property='value'),
     Input(component_id='steam-weekend-checkbox', component_property='value')]
)
def steam_graph(data, sch, weekendSplit):
    fig, savings = Energy_bar_graph(data, sch, weekendSplit, 'steam')
    return fig, savings


@app.callback(
    [Output(component_id='water-energy-graph', component_property='figure'),
     Output(component_id='water-data-est', component_property='children')],
    [Input(component_id='data-est', component_property='children'),
     Input(component_id='water-energy-slider', component_property='value'),
     Input(component_id='water-weekend-checkbox', component_property='value')]
)
def steam_graph(data, sch, weekendSplit):
    fig, savings = Energy_bar_graph(data, sch, weekendSplit, 'chilledWater')
    return fig, savings


@app.callback(
    [Output(component_id='elec-energy-graph', component_property='figure'),
     Output(component_id='elec-data-est', component_property='children')],
    [Input(component_id='data-est', component_property='children'),
     Input(component_id='elec-energy-slider', component_property='value'),
     Input(component_id='elec-weekend-checkbox', component_property='value')]
)
def steam_graph(data, sch, weekendSplit):
    fig, savings = Energy_bar_graph(data, sch, weekendSplit, 'electricity')
    return fig, savings


# ------------------- Global energy slider
@app.callback(
    [Output(component_id='steam-energy-slider', component_property='value'),
     Output(component_id='water-energy-slider', component_property='value'),
     Output(component_id='elec-energy-slider', component_property='value'),
     Output(component_id='steam-weekend-checkbox', component_property='value'),
     Output(component_id='water-weekend-checkbox', component_property='value'),
     Output(component_id='elec-weekend-checkbox', component_property='value'),
     Output(component_id='steam-energy-slider', component_property='style'),
     Output(component_id='water-energy-slider', component_property='style'),
     Output(component_id='elec-energy-slider', component_property='style'),
     Output(component_id='steam-weekend-checkbox', component_property='style'),
     Output(component_id='water-weekend-checkbox', component_property='style'),
     Output(component_id='elec-weekend-checkbox', component_property='style'),
     Output(component_id='steam-slider-text', component_property='style'),
     Output(component_id='water-slider-text', component_property='style'),
     Output(component_id='elec-slider-text', component_property='style'),
     Output(component_id='main-energy-slider', component_property='style'),
     Output(component_id='main-weekend-checkbox', component_property='style')],
    [Input(component_id='main-energy-slider', component_property='value'),
     Input(component_id='main-weekend-checkbox', component_property='value'),
     Input(component_id='adjust-individually-checkbox', component_property='value')]
)
def global_slider(main_slider, weekend_checkbox, main_checkbox):
    if main_checkbox==[True]:
        individual_display = {'display': 'block'}
        main_display = {'display': 'None'}
    else:
        individual_display = {'display': 'None'}
        main_display = {'display': 'block'}
    return main_slider, main_slider, main_slider,\
           weekend_checkbox, weekend_checkbox, weekend_checkbox,\
           individual_display, individual_display, individual_display, \
           individual_display, individual_display, individual_display, \
           individual_display, individual_display, individual_display, \
           main_display, main_display

@app.callback(
    [Output(component_id='summary-text', component_property='children')],
    [Input(component_id='steam-data-est', component_property='children'),
     Input(component_id='water-data-est', component_property='children'),
     Input(component_id='elec-data-est', component_property='children')]
)
def output_text(steam_data, water_data, elec_data):
    steam_savings = steam_data[1] / steam_data[0] * 100
    water_savings = water_data[1] / water_data[0] * 100
    elec_savings = elec_data[1] / elec_data[0] * 100
    total_savings = (steam_data[1] + water_data[1] + elec_data[1]) / (steam_data[0] + water_data[0] + elec_data[0]) * 100

    def format_text(savings, parameter):
        if savings > 101:
            text_estimate = f'Increases {parameter} energy by {savings - 100:.0f}%'
        elif savings < 99:
            text_estimate = f'Decreases {parameter} energy by {100 - savings:.0f}%'
        else:
            text_estimate = f'Has no impact on {parameter} energy'
        return text_estimate

    text = [html.Br(),
            f'Your proposed Schedule:', html.Br(),
            format_text(total_savings, 'Total'), html.Br(),
            format_text(steam_savings, 'heating'), html.Br(),
            format_text(water_savings, 'cooling'), html.Br(),
            format_text(elec_savings, 'electrical'), html.Br()]

    return [text]