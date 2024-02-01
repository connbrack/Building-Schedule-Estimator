from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
import pathlib
import pickle as plk

from app import app
from apps._fun import schedule_seperate_graph, LinModel, detect_schedule

layout = dbc.Container([
    # ---------- Hidden Values
    html.Div(id='data_sch', style={'display': 'none'}),
    html.Div(id='page_load_trigger_sch', style={'display': 'none'}),


    html.Div(id='steam-line-sch', style={'display': 'none'}, children=[]),
    html.Div(id='water-line-sch', style={'display': 'none'}, children=[]),
    html.Div(id='elec-line-sch', style={'display': 'none'}, children=[]),
    # -----------------

    dbc.Row(
        html.H1('Find and Model Schedule'),
    ),

    dbc.Row([
        dbc.Col([
            html.P("Building",
                   className="font-weight-bolder"),
            dcc.Dropdown(
                id='building_dropdown_sch', value=[], clearable=False,
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
                id='year_dropdown_sch', value=2020, clearable=False,
                persistence=True, persistence_type='session',
                options=[],
            )
        ]),

        dbc.Col([])
    ]),

    dbc.Row([
        dbc.Col([
            html.Br(),
            html.H6("Steam"),
            dcc.Graph(id='steam_graph', figure={}),
            dbc.Button('Automatically Detect Schedule', color='primary', id='steam-detect-button', className="me-1"),
            html.Br(),
            html.P("Manually Adjust Schedule",
                   className="font-weight-bolder"),
            dcc.RangeSlider(id='steam-slider',
                            min=0, max=24, step=1, value=[5, 17], pushable=2,
                            tooltip={"placement": "bottom", "always_visible": True}
                            ),
            html.Br()

        ]),
        dbc.Col([
            html.Br(),
            html.H6("Chilled Water"),
            dcc.Graph(id='water_graph', figure={}),
            dbc.Button("Automatically Detect Schedule", id='water-detect-button', color="primary"),
            html.Br(),
            html.P("Manually Adjust Schedule",
                   className="font-weight-bolder"),
            dcc.RangeSlider(id='water-slider',
                            min=0, max=24, step=1, value=[5, 17], pushable=2,
                            tooltip={"placement": "bottom", "always_visible": True}
                            ),
            html.Br()
        ]),

        dbc.Col([
            html.Br(),
            html.H6("Electricity"),
            dcc.Graph(id='elec_graph', figure={}),
            dbc.Button("Automatically Detect Schedule", id='elec-detect-button', color="primary"),
            html.Br(),
            html.P("Manually Adjust Schedule",
                   className="font-weight-bolder"),
            dcc.RangeSlider(id='elec-slider',
                            min=0, max=24, step=1, value=[5, 17], pushable=2,
                            tooltip={"placement": "bottom", "always_visible": True}
                            ),
            html.Br()

        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.Br(), html.Br(), html.Br(),
            dbc.Button("Accept Schedule and train model", id='train-model-button', color="primary"),
            html.Div(id='sch-status-text', children=[])
        ])
    ])

], fluid=True)

@app.callback(
    [Output(component_id='building_dropdown_sch', component_property='value'),
     Output(component_id='year_dropdown_sch', component_property='value')],
    [Input(component_id='page_load_trigger_sch', component_property='children')]
)
def page_load_trigger_sch(t):
    PATH = pathlib.Path(__file__).parent
    DATA_PATH = PATH.joinpath("../saved").resolve()

    with open(DATA_PATH.joinpath(f'model.pkl'), 'rb') as f:
        model = plk.load(f)

    building, year, steam_sch, _, _, _, water_sch, _, _, _, elec_sch, _, _, _ = model

    return building, year

@app.callback(
    [Output(component_id='data_sch', component_property='children'),
     Output(component_id='year_dropdown_sch', component_property='options')],
    [Input(component_id='building_dropdown_sch', component_property='value'),
     Input(component_id='year_dropdown_sch', component_property='value')]
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

    # Convert to Json for storage
    data = df.to_json(date_format='iso', orient='split')

    return data, options

# Schedule detect
@app.callback(
    [Output(component_id='steam-slider', component_property='value')],
    [Input(component_id='steam-detect-button', component_property='n_clicks'),
     Input(component_id='data_sch', component_property='children')]
)
def steam_sch_detect(n_clicks, data):
    if n_clicks is not None:
        data = pd.read_json(data, orient='split')
        sch = detect_schedule(data, 'steam', model=2, weekend_split=True)
    else:
        PATH = pathlib.Path(__file__).parent
        DATA_PATH = PATH.joinpath("../saved").resolve()

        with open(DATA_PATH.joinpath(f'model.pkl'), 'rb') as f:
            model = plk.load(f)
        _, _, steam_sch, _, _, _, _, _, _, _, _, _, _, _ = model
        sch = steam_sch

    return [sch]


@app.callback(
    [Output(component_id='water-slider', component_property='value')],
    [Input(component_id='water-detect-button', component_property='n_clicks'),
     State(component_id='data_sch', component_property='children')]
)
def water_sch_detect(n_clicks, data):
    if n_clicks is not None:
        data = pd.read_json(data, orient='split')
        sch = detect_schedule(data, 'chilledWater', model=2, weekend_split=True)
    else:
        PATH = pathlib.Path(__file__).parent
        DATA_PATH = PATH.joinpath("../saved").resolve()

        with open(DATA_PATH.joinpath(f'model.pkl'), 'rb') as f:
            model = plk.load(f)
        _, _, _, _, _, _, water_sch, _, _, _, _, _, _, _ = model
        sch = water_sch

    return [sch]

@app.callback(
    [Output(component_id='elec-slider', component_property='value')],
    [Input(component_id='elec-detect-button', component_property='n_clicks'),
     Input(component_id='data_sch', component_property='children')]
)
def elec_sch_detect(n_clicks, data):
    if n_clicks is not None:
        data = pd.read_json(data, orient='split')
        sch = detect_schedule(data, 'electricity', model=2, weekend_split=True)
    else:
        PATH = pathlib.Path(__file__).parent
        DATA_PATH = PATH.joinpath("../saved").resolve()

        with open(DATA_PATH.joinpath(f'model.pkl'), 'rb') as f:
            model = plk.load(f)
        _, _, _, _, _, _, _, _, _, _, elec_sch, _, _, _ = model
        sch = elec_sch

    return [sch]


# ------------------------- Figures
@app.callback(
    [Output(component_id='steam_graph', component_property='figure')],
    [Input(component_id='data_sch', component_property='children'),
     Input(component_id='steam-slider', component_property='value'),
     Input(component_id='building_dropdown_sch', component_property='value'),
     Input(component_id='year_dropdown_sch', component_property='value'),
     Input(component_id='sch-status-text', component_property='children')]
)
def steam_schedule_separate(data, sch, building, year, update):
    data = pd.read_json(data, orient='split')
    fig = schedule_seperate_graph(building, year, data, sch, 'steam', 'red', 'LightCoral', None, 'black', 'grey', weekend_included=True)
    return [fig]

@app.callback(
    [Output(component_id='water_graph', component_property='figure')],
    [Input(component_id='data_sch', component_property='children'),
     Input(component_id='water-slider', component_property='value'),
     Input(component_id='building_dropdown_sch', component_property='value'),
     Input(component_id='year_dropdown_sch', component_property='value'),
     Input(component_id='sch-status-text', component_property='children')]
)
def water_schedule_separate(data, sch, building, year, update):
    data = pd.read_json(data, orient='split')
    fig = schedule_seperate_graph(building, year, data, sch, 'chilledWater', 'blue', 'LightSkyBlue', None, 'black', 'grey', weekend_included=True)
    return [fig]

@app.callback(
    [Output(component_id='elec_graph', component_property='figure')],
    [Input(component_id='data_sch', component_property='children'),
     Input(component_id='elec-slider', component_property='value'),
     Input(component_id='building_dropdown_sch', component_property='value'),
     Input(component_id='year_dropdown_sch', component_property='value'),
     Input(component_id='sch-status-text', component_property='children')]
)
def elec_schedule_separate(data, sch, building, year, update):
    data = pd.read_json(data, orient='split')
    fig = schedule_seperate_graph(building, year, data, sch, 'electricity', 'black', 'grey', None, 'red', 'LightCoral', weekend_included=True)
    return [fig]

# ------------------------- Figures


@app.callback(
    [Output(component_id='sch-status-text', component_property='children')],
    [Input(component_id='train-model-button', component_property='n_clicks'),
     State(component_id='data_sch', component_property='children'),
     State(component_id='building_dropdown_sch', component_property='value'),
     State(component_id='year_dropdown_sch', component_property='value'),
     State(component_id='steam-slider', component_property='value'),
     State(component_id='water-slider', component_property='value'),
     State(component_id='elec-slider', component_property='value'),
     State(component_id='steam-line-sch', component_property='children'),
     State(component_id='water-line-sch', component_property='children'),
     State(component_id='elec-line-sch', component_property='children')]
)
def train_steam_model(n, data, building, year, steam_sch, water_sch, elec_sch, steam_line, water_line, elec_line):
    if n is not None:
        data = pd.read_json(data, orient='split')
        steam_model_on, steam_model_off, steam_model_r2 = LinModel(data, steam_sch, 'steam', model=2, WeekendSplit=True)
        water_model_on, water_model_off, water_model_r2 = LinModel(data, water_sch, 'chilledWater', model=2, WeekendSplit=True)
        elec_model_on, elec_model_off, elec_model_r2 = LinModel(data, elec_sch, 'electricity', model=3, WeekendSplit=True)

        model = [building, year,
                 steam_sch, steam_model_on, steam_model_off, steam_model_r2,
                 water_sch, water_model_on, water_model_off, water_model_r2,
                 elec_sch, elec_model_on, elec_model_off, elec_model_r2]

        PATH = pathlib.Path(__file__).parent
        DATA_PATH = PATH.joinpath("../saved").resolve()

        with open(DATA_PATH.joinpath(f'model.pkl'), 'wb') as f:
            plk.dump(model, f)

    else:
        PATH = pathlib.Path(__file__).parent
        DATA_PATH = PATH.joinpath("../saved").resolve()

        with open(DATA_PATH.joinpath(f'model.pkl'), 'rb') as f:
            model = plk.load(f)

        building, year, steam_sch, _, _, steam_model_r2, water_sch, _, _, water_model_r2, elec_sch, _, _, elec_model_r2 = model

    text = [html.Br(),
            f'You have trained a model based on the following parameters:', html.Br(),
            f'Building: {building}, Year: {year}', html.Br(),
            f'Heating schedule from {steam_sch[0]}:00 to {steam_sch[1]}:00', html.Br(),
            f'Cooling schedule from {water_sch[0]}:00 to {water_sch[1]}:00', html.Br(),
            f'Electricity schedule from {elec_sch[0]}:00 to {elec_sch[1]}:00', html.Br()]

    return [text]
