import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Input
from dash.dependencies import Output

import dash
import math
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}


class GraphPlot:
    def __init__(self, df, review_count, keras_acc_fig, keras_loss_fig, spark_acc_fig):
        self.default_df = df
        self.df = df
        self.review_count = review_count
        self.keras_acc_fig = keras_acc_fig
        self.keras_loss_fig = keras_loss_fig
        self.spark_acc_fig = spark_acc_fig
        self.app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
        self.setup_layout()
        self.setup_callbacks()
        self.app.run_server(debug=False)

    def setup_layout(self):
        self.app.layout = html.Div(children=[
            html.H1(children='DEDS Assignment 2 Dashboard'),
            dcc.Tabs(id="tabs-styled-with-inline", value='tab-1', children=[
                dcc.Tab(label='Map', value='tab-1', style=tab_style,
                        selected_style=tab_selected_style),
                dcc.Tab(label="Compare", value='tab-2', style=tab_style,
                        selected_style=tab_selected_style),
                dcc.Tab(label="Statistics", value='tab-3', style=tab_style,
                        selected_style=tab_selected_style),
                dcc.Tab(label="Machine/Deep learning solutions", value='tab-4', style=tab_style,
                        selected_style=tab_selected_style),
            ], style=tabs_styles),
            html.Div(id='tabs-content-inline')
        ], style={'columnCount': 1})

    def setup_callbacks(self):
        def get_hotel_count_fig():
            self.review_count.amount_reviews = self.review_count.amount_reviews.astype(int)
            top_hotels = self.review_count.nlargest(15, columns=['amount_reviews'])
            fig = px.bar(top_hotels, x="hotel", y="amount_reviews", title="Top 15 most reviewed hotels")

            return fig

        @self.app.callback(Output('best-rating-graph', 'figure'),
                           Output('worst-rating-graph', 'figure'),
                           Input('hotels-area', 'value'))
        def get_rating_graph(hotels_area):
            filtered_hotels = self.df
            filtered_hotels['lng'] = filtered_hotels['lng'].astype(float)
            filtered_hotels['lat'] = filtered_hotels['lat'].astype(float)
            if hotels_area == "Amsterdam":
                filtered_hotels = filtered_hotels.loc[
                    (filtered_hotels["lng"] > 4.8) & (filtered_hotels["lng"] < 5.0) & (
                            filtered_hotels["lat"] > 52.2) & (
                            filtered_hotels["lat"] < 52.45)]
            elif hotels_area == "Barcelona":
                filtered_hotels = filtered_hotels.loc[
                    (filtered_hotels["lng"] > 2.1) & (filtered_hotels["lng"] < 2.3) & (
                            filtered_hotels["lat"] > 41.3) & (
                            filtered_hotels["lat"] < 41.45)]
            elif hotels_area == "Vienna":
                filtered_hotels = filtered_hotels.loc[
                    (filtered_hotels["lng"] > 16.25) & (filtered_hotels["lng"] < 16.45) & (
                            filtered_hotels["lat"] > 48.1) & (
                            filtered_hotels["lat"] < 48.3)]
            elif hotels_area == "London":
                filtered_hotels = filtered_hotels.loc[
                    (filtered_hotels["lng"] > -0.4) & (filtered_hotels["lng"] < 0.1) & (
                            filtered_hotels["lat"] > 51.4) & (
                            filtered_hotels["lat"] < 51.7)]
            elif hotels_area == "Paris":
                filtered_hotels = filtered_hotels.loc[
                    (filtered_hotels["lng"] > 2.2) & (filtered_hotels["lng"] < 2.45) & (
                            filtered_hotels["lat"] > 48.8) & (
                            filtered_hotels["lat"] < 49)]
            elif hotels_area == "Milan":
                filtered_hotels = filtered_hotels.loc[
                    (filtered_hotels["lng"] > 9.15) & (filtered_hotels["lng"] < 9.25) & (
                            filtered_hotels["lat"] > 45.4) & (
                            filtered_hotels["lat"] < 45.5)]

            filtered_hotels['Average_Score'] = filtered_hotels['Average_Score'].astype(float)
            top_hotels = filtered_hotels.nlargest(15, columns=['Average_Score'])
            worst_hotels = filtered_hotels.nsmallest(15, columns=['Average_Score'])

            best_score_fig = px.bar(top_hotels, x="Hotel_Name", y="Average_Score",
                                    title="Top 15 best rated hotels ")
            best_score_fig.update_layout(yaxis_range=[math.floor(top_hotels["Average_Score"].min()),
                                                      math.ceil(top_hotels["Average_Score"].max())])
            worst_score_fig = px.bar(worst_hotels, x="Hotel_Name", y="Average_Score",
                                     title="Top 15 worst rated hotels ")
            worst_score_fig.update_layout(yaxis_range=[math.floor(worst_hotels["Average_Score"].min()),
                                                       math.ceil(worst_hotels["Average_Score"].max())])
            return best_score_fig, worst_score_fig

        @self.app.callback(Output('map-graph', 'figure'),
                           Input('score-slider', 'value'))
        def get_map_fig(rating_range):
            px.set_mapbox_access_token(open("assets/mapbox_token").read())

            # score
            bool_series = self.df['Average_Score'].astype(float).between(rating_range[0], rating_range[1])
            filtered_df = self.df[bool_series]
            fig = go.Figure()

            fig.add_trace(go.Scattermapbox(
                lat=filtered_df.lat,
                lon=filtered_df.lng,
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=10,
                    color='rgb(255, 0, 0)',
                    opacity=0.7
                ),
                hoverinfo=['all'],
                text="Hotel name: <b>" + filtered_df.Hotel_Name.astype(str) +
                     "</b><br>Average score: " + filtered_df.Average_Score.astype(str),
            ))

            fig.update_layout(
                height=600,
                autosize=True,
                hovermode='closest',
                mapbox=dict(
                    style='open-street-map',
                    bearing=0,
                    pitch=0,
                    center=dict(
                        lat=49.724479,
                        lon=7.884550
                    ),
                    zoom=3
                ),
            )
            return fig

        @self.app.callback(Output('tabs-content-inline', 'children'),
                           Input('tabs-styled-with-inline', 'value'))
        def render_content(tab):
            if tab == 'tab-1':
                return html.Div([
                    dcc.Graph(id="map-graph"),
                    html.H4("Average hotel score", style={"text-align": "center"}),
                    html.Div([
                        dcc.RangeSlider(
                            id="score-slider",
                            min=0,
                            max=10,
                            step=None,
                            marks={
                                0: '0',
                                1: '1',
                                2: '2',
                                3: '3',
                                4: '4',
                                5: '5',
                                6: '6',
                                7: '7',
                                8: '8',
                                9: '9',
                                10: '10',
                            },
                            value=[0, 10],
                        ),
                    ], style={"padding": "0px 185px 25px"})
                ])
            elif tab == 'tab-2':
                div = html.Div([
                    dcc.Dropdown(
                        id='hotels-area',
                        options=[
                            {'label': 'Amsterdam', 'value': 'Amsterdam'},
                            {'label': 'Barcelona', 'value': 'Barcelona'},
                            {'label': 'London', 'value': 'London'},
                            {'label': 'Milan', 'value': 'Milan'},
                            {'label': 'Paris', 'value': 'Paris'},
                            {'label': 'Vienna', 'value': 'Vienna'}
                        ]
                    ),
                    html.Div([
                        dcc.Graph(id="best-rating-graph"),
                        dcc.Graph(id="worst-rating-graph")
                    ], style={"columnCount": 2})
                ], style={"margin": "25px 0 0 0"})
                return div
            elif tab == 'tab-3':
                fig = get_hotel_count_fig()
                return html.Div([
                    dcc.Graph(figure=fig)
                ])
            elif tab == 'tab-4':
                return html.Div([
                    html.H4("Keras classification", style={"text-align": "center"}),
                    html.Div([
                        dcc.Graph(figure=self.keras_acc_fig),
                        dcc.Graph(figure=self.keras_loss_fig)
                    ], style={'columnCount': 2}),
                    html.H4("Spark classification", style={"text-align": "center"}),
                    html.Div([
                        dcc.Graph(figure=self.spark_acc_fig),
                    ], style={"width": "800px"})
                ])
