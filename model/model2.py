import torch
from neuralprophet import NeuralProphet
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os


def ARnet(section1, section2, count, save, period=4):

    try:
        if section2 in section1.columns[1:]:
            df = pd.DataFrame({
                "ds": section1.iloc[:, 1],
                "y": section1[section2]
            })
            m = NeuralProphet(loss_func=torch.nn.L1Loss)
            df_train, df_test = m.split_df(df, freq="M", valid_p=0.1)
            metrics = m.fit(df_train, freq="M", validation_df=df_test)
            metrics.tail(1)

            pred_train = m.predict(df)
            pred_test = m.predict(df_test)
            #prediction over the next period = "default:4" months
            next_dataset = m.make_future_dataframe(df_test, periods=period)
            pred_next = m.predict(df=next_dataset)

            #plot

            df = pd.concat([pred_train, pred_next])
            final = go.Figure()
            final.add_trace(
                go.Scatter(x=df['ds'][:36],
                           y=df['yhat1'][:36],
                           mode='lines',
                           name='Training set'))
            final.update_layout(title={
                'text':
                f"Predicción del consumo mensual(Litro/Mes) en {section2}",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
                                xaxis_title="Meses",
                                yaxis_title="Consumo(Litro/Mes)")
            final.add_trace(
                go.Scatter(x=df['ds'][35:35 + period + 1],
                           y=df['yhat1'][35:35 + period + 1],
                           mode='lines',
                           name='Prediction'))
            final.add_trace(
                go.Scatter(x=df['ds'],
                           y=df['y'],
                           mode='markers',
                           name='Actual',
                           line_color="firebrick"))
            final.add_trace(
                go.Scatter(x=df['ds'][33:36],
                           y=df['yhat1'][33:36],
                           mode='lines',
                           name='test set',
                           line_color="#32CD32"))

            fig_param = m.plot_parameters(plotting_backend="plotly")
            fig_param.update_layout(
                title={
                    'text':
                    f"Análisis de la tendencia del consumo en {section2}",
                    'y': 0.99,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {
                        'size': 16
                    },
                })
            #1

            path = "./model/images/"
            if count == 0:
                path += "comercial/"
            elif count == 1:
                path += "domestico/"
            elif count == 2:
                path += "industrial/"

            path += section2 + "/"

            if not os.path.exists(path):
                os.makedirs(path)

            prediction_image = path + "prediction.png"
            anlisi_image = path + "analisi.png"
            final.write_image(prediction_image)
            fig_param.write_image(anlisi_image)

            return m, pred_train, pred_next, fig_param

    except Exception as e:
        pass


data_location = [
    "./model/data/COMER_NORM_resum.xlsx",
    "./model/data/DOMESTICO_NORM_resum.xlsx",
    "./model/data/INDUS_NORM_resum.xlsx"
]
#original = pd.read_excel(data_location)
#original = original.iloc[1:, :]
# model,pre

zonas = [
    "BADALONA", "BARCELONA", "BEGUES", "CASTELLDEFELS", "CERDANYOLA",
    "CORNELLA", "EL PAPIOL", "ESPLUGUES", "GAVA", "L'HOSPITALET LLOBR.",
    "LA LLAGOSTA", "LES BOTIGUES SITGES", "MONTCADA I REIXAC", "MONTGAT",
    "PALLEJA", "SANT ADRIA", "SANT BOI", "SANT CLIMENT LLOB.", "SANT CUGAT",
    "SANT FELIU LL.", "SANT JOAN DESPI", "SANT JUST DESVERN",
    "STA.COLOMA CERVELLO", "STA.COLOMA GRAMENET", "TORRELLES LLOBREGAT",
    "VILADECANS"
]

count = 0
for data in data_location:
    original = pd.read_excel(data)
    original = original.iloc[1:, :]
    print(data)
    for zona in zonas:
        print(zona)
        if zona in original.columns:
            m, pred_train, pred_next, fig_param = ARnet(original,
                                                        zona,
                                                        count,
                                                        save="True")

    count += 1
