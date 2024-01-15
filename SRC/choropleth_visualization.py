#pip install dash
#pip install dash-bootstrap-components
from dash import Dash, html, dcc, Input, Output, State
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import math



# Reading sample data using pandas DataFrame
csvfile = '../data/clean_agg_set.csv'
df = pd.read_csv(csvfile)




g = list(df['week'].unique())
g.sort()
smallest = min(g)
biggest = max(g)


max_step = 40
qty_time = len(g) + smallest
step_required = math.ceil(qty_time/max_step)

mark_dict={}
for i in range(int(smallest), int(biggest), int(step_required)):
    d_list = df[df['week'] == i]['true_week_time'].to_list()
    if i == 50:
        mark_dict[i] = {'label':'2020-12-16',"style": {"transform": "rotate(45deg)"}}
    if i == 102:
        mark_dict[i] = {'label': '2021-12-29', "style": {"transform": "rotate(45deg)"}}
    if len(d_list) > 0:
        mark_dict[i] = {'label':d_list[0],"style": {"transform": "rotate(45deg)"}}


# Building components
app = Dash(__name__, external_stylesheets=[dbc.themes.LUMEN])

mytitle = dcc.Markdown(children='',style={"textAlign": "center"})

mygraph = dcc.Graph(id="my-choropleth",
                    figure={})

dropdown = dcc.Dropdown(id="my-dropdown",
                        options=df.columns.values[4:8],
                        value='Sentiment Score',  # initial value displayed when page first loads
                        clearable=False,
                        style={'margin-top':"30px"})

label = dbc.Label("Select Weeks:",
                  className="fw-bold",
                  style={"textDecoration": "underline", "fontSize": 20} )

# spacer = dcc.Input(style={'margin-top':"15px"})

slider = dcc.RangeSlider(id="my-slider",
                         min=smallest,
                         max=biggest,
                         step=1,
                         # value=[smallest, smallest+1],
                         value=[9, 9],
                         marks=mark_dict,
                         # marks={i: str(i) for i in range(int(smallest), int(biggest), int(step_required))},
                         )

button = dbc.Button(id="my-button",
                    children="Submit",
                    n_clicks=0,
                    color="primary",
                    className="mt-4",
                    )

# Creating Layout
app.layout = dbc.Container(
    [
        html.H1(
            "Choropleth to Visualize Sentiment Analysis of COVID-19 Tweets Worldwide",
            style={"textAlign": "center"},
        ),
        dbc.Row([
            dbc.Col(
                [
                    mytitle,
                    mygraph,
                ],
                width=12,
            )
        ]),
        dbc.Row(
            [
                dbc.Col(
                    [
                        label,
                        slider,
                        # spacer,
                        dropdown,
                        button
                    ],
                    width=10,
                ),
            ]
        )
    ]
)

# Callback so components can interact
@app.callback(
    Output("my-choropleth", "figure"),
    Output(mytitle, 'children'),
    Input("my-button", "n_clicks"),
    Input("my-dropdown", "value"),
    State("my-slider", "value"),
)
def update_graph(n_clicks, indct_chosen, slider_value):
    dff = pd.read_csv(csvfile)

    if slider_value[0] != slider_value[1]:
        dff = dff[dff.week.between(slider_value[0], slider_value[1])]
        dff = dff.groupby(["code", "country"])[indct_chosen].mean()
        dff = dff.reset_index()


        fig = px.choropleth(
            data_frame=dff,
            locations="code",
            color=indct_chosen,
            scope="world",
            color_continuous_scale="thermal",
            hover_data={"code": False, "country": True},
        )
        fig.update_layout(
            geo={"projection": {"type": "natural earth"}},
            margin=dict(l=50, r=50, t=20, b=10),
        )
        # fig.update_geos(bgcolor= 'rgba(0,0,0,0)')
        return fig, '## '+indct_chosen

    if slider_value[0] == slider_value[1]:
        dff = dff[dff["week"].isin(slider_value)]

        fig = px.choropleth(
            data_frame=dff,
            locations="code",
            color=indct_chosen,
            scope="world",
            color_continuous_scale="thermal",
            hover_data={"code": False, "country": True},
        )
        fig.update_layout(
            geo={"projection": {"type": "natural earth"}},
            margin=dict(l=50, r=50, t=20, b=10),
        )
        # fig.update_geos(bgcolor= 'rgba(0,0,0,0)')
        return fig, '## '+indct_chosen


if __name__ == "__main__":
    app.run_server(debug=True)

