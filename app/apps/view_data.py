from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import pathlib
from app import app
import pickle as plk


layout = dbc.Container([
    # ---------- Hidden Values
    html.Div(id='BuildingDataHolder', style={'display': 'none'}),
    # -----------------

    dbc.Row(
        html.H1('View Data'),
    ),

    dbc.Row([
        dbc.Col([
            html.P("Building",
                   className="font-weight-bolder"),
            dcc.Dropdown(
                id='building_dropdown', value='CD Howe', clearable=False,
                persistence=True, persistence_type='session',
                options=[
                    {'label': 'CD Howe', 'value': 'CD Howe'},
                    {'label': 'East Memorial', 'value': 'East Memorial'},
                    {'label': 'Saint Andrews Tower', 'value': 'Saint Andrews Tower'}
                ]
            ),
        ]),

        dbc.Col([
            html.P("Year",
                   className="font-weight-bolder"),
            dcc.Dropdown(
                id='year_dropdown', value=2020, clearable=False,
                persistence=True, persistence_type='session',
                options=[],
            )
        ]),

        dbc.Col([
            html.P("Graph",
                   className="font-weight-bolder"),
            dcc.Dropdown(
                id='graph_dropdown', value='time', clearable=False,
                persistence=True, persistence_type='session',
                options=[
                    {'label': 'Standard', 'value': 'time'},
                    {'label': 'Outdoor Air Temperature', 'value': 'OAT'}
                ]
            )
        ])
    ]),

    dcc.Graph(id='chPt_plot', figure={})

], fluid=True)


@app.callback(
    [Output(component_id='BuildingDataHolder', component_property='children'),
     Output(component_id='year_dropdown', component_property='options')],
    [Input(component_id='building_dropdown', component_property='value'),
     Input(component_id='year_dropdown', component_property='value')]
)
def update_Dataset(building_dropdown, year_dropdown):
    ## Ged data sets
    PATH = pathlib.Path(__file__).parent
    DATA_PATH = PATH.joinpath("../data").resolve()

    with open(DATA_PATH.joinpath(f'{building_dropdown}/Energy/_energyData.pkl'), 'rb') as f:
        energy_df = plk.load(f)

    with open(DATA_PATH.joinpath(f'Supporting/Weather/Ottawa.pkl'), 'rb') as f:
        weather_df = plk.load(f)

    # Get date range of dataset
    years = energy_df.index.year.unique().values
    options = [{'label': str(y), 'value': y} for y in years]


    # Combined to single dataframe
    df = pd.merge(energy_df, weather_df, how='outer', left_index=True, right_index=True)
    df = df[df.index.year == year_dropdown]

    #Convert to Json for storage
    data = df.to_json(date_format='iso', orient='split')

    return data, options

@app.callback(
    Output(component_id='chPt_plot', component_property='figure'),
    [Input(component_id='BuildingDataHolder', component_property='children'),
    Input(component_id='graph_dropdown', component_property='value')]
)
def plot_data(BuildingDataHolder, graph_dropdown):
    data = pd.read_json(BuildingDataHolder, orient='split')

    if graph_dropdown == 'time':
        fig = go.Figure()
        fig.add_scatter(name='Electricity', x=data.index, y=data['electricity'],
                        customdata=data.index, marker=dict(color='darkgray'))
        fig.add_scatter(name='Steam', x=data.index, y=data['steam'],
                        customdata=data.index, marker=dict(color='red'))
        fig.add_scatter(name='Chilled Water', x=data.index, y=data['chilledWater'],
                    customdata=data.index, marker=dict(color='blue'))
        fig.update_layout(template="none", yaxis_title="Energy (MJ)", height=600)

    else:
        fig = go.Figure()
        fig.add_scatter(name='Electricity', x=data.temperature, y=data['electricity'],
                        customdata=data.index, mode='markers', marker=dict(color='darkgray'))
        fig.add_scatter(name='Steam', x=data.temperature, y=data['steam'],
                        customdata=data.index, mode='markers', marker=dict(color='red'))
        fig.add_scatter(name='Chilled Water', x=data.temperature, y=data['chilledWater'],
                        customdata=data.index, mode='markers', marker=dict(color='blue'))
        fig.update_layout(xaxis_title="Outdoor Temperature")
        fig.update_layout(template="none", yaxis_title="Energy per hour (MJ/hr)", height=600)

    return fig
