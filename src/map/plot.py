from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import json

df = pd.read_excel(r".\datos\sum_consumo_total.xlsx")
file = open(r".\datos\map.geojson",encoding='utf-8')
municipios = json.load(file)
op={'Doméstico':'DOMESTICO','Industrial':'INDUSTRIAL','Comercial':'COMERCIAL'}

app = Dash(__name__)

app.layout = html.Div([
    html.H4('Consumo mensual'),
    html.P("Seleccionar el tipo de consumo:"),
    dcc.RadioItems(
        id='tipo', 
        options=["Doméstico", "Industrial", "Comercial"],
        value="Doméstico",
        inline=True
    ),
    dcc.Graph(id='graf')
])

@app.callback(
    Output("graf", "figure"), 
    Input("tipo", "value"))
def display_choropleth(type):
    df_filt = df.loc[df['TIPO']==op[type]]
    fig = px.choropleth_mapbox(df_filt, geojson=municipios, featureidkey='properties.CODIGO_INE',locations='CODIGO_INE', color='SUM_CONSUMO',
                            range_color=(0, df_filt['SUM_CONSUMO'].max()),
                            hover_name='MUNICIPIO',
                            color_continuous_scale="darkmint",
                            animation_frame='FECHA',
                            animation_group='MUNICIPIO',
                            mapbox_style="carto-positron",
                            zoom=9.5, center = {"lat": 41.3568577, "lon": 2.0710342},
                            opacity=0.5,
                            labels={'SUM_CONSUMO':'Suma de consumo'}
                            )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig


app.run_server()

# openpyxl.
# dash 
# plotly
