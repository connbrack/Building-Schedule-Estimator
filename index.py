import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

# Connect to main app.py file
from app import app
from app import server

# Connect to your app pages
from apps import view_data, schedule_finder, energy_estimator

# styling the sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "22rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "25rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Setback Schedule Estimator", className="display-4"),
        html.Hr(),
        html.P(
            "Use this tool to discover the setback schedule in your building and estimate the impact of changes", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("View Data", href="/", active="exact"),
                dbc.NavLink("Find Schedule", href="/schedule_finder", active="exact"),
                dbc.NavLink("Energy Estimator", href="/energy_estimator", active="exact")
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)

app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content,

    # Persistant Variables
    dcc.Store(id='building_dropdown_selection', data=None),
    dcc.Store(id='year_dropdown_selection', data=None),
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return view_data.layout
    if pathname == '/schedule_finder':
        return schedule_finder.layout
    if pathname == '/energy_estimator':
        return energy_estimator.layout
    else:
        return "404 Page Error! Please choose a link"


if __name__ == '__main__':
    app.run_server(debug=False)