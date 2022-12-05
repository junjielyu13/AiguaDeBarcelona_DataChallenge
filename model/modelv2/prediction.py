#pip install neuralprophet

import os
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import torch
import kaleido
from neuralprophet import NeuralProphet


def ARnet(data_location,
          section2,
          period=4,
          save_prediction=False,
          save_img=False,
          save_model=False,
          show=False):

    section1 = pd.read_excel(data_location)
    consumo = data_location.split("\\")[-1][:-3]
    mensual = consumo.split("_")[1] == "mensual"
    actividad = consumo.split("_")[2][:-2]

    if mensual == True:
        df = pd.DataFrame({"ds": section1.iloc[:, 1], "y": section1[section2]})
        m = NeuralProphet(loss_func=torch.nn.L1Loss)
        df_train, df_test = m.split_df(df, freq="M", valid_p=0.1)
        metrics = m.fit(df_train, freq="M", validation_df=df_test)
    else:
        df = pd.DataFrame({"ds": section1["FECHA"], "y": section1[section2]})
        m = NeuralProphet(loss_func=torch.nn.L1Loss)
        df_train, df_test = m.split_df(df, freq="D", valid_p=0.1)
        metrics = m.fit(df_train, freq="D", validation_df=df_test)

    cost = metrics.loc[metrics.shape[0] - 1, "RMSE"]
    pred_train = m.predict(df)
    next_dataset = m.make_future_dataframe(df_test, periods=period)
    pred_next = m.predict(df=next_dataset)
    df = pd.concat([pred_train, pred_next])

    if save_prediction == True:
        path = "./model/modelv2/prediction/"
        if not os.path.exists(path):
            os.makedirs(path)
        df.to_csv(path + f"result_{actividad}_{section2}.csv")

    if mensual == True:
        final = go.Figure()
        final.add_trace(
            go.Scatter(x=df['ds'],
                       y=df["yhat1"],
                       mode='lines',
                       name='Training set'))
        final.update_layout(title={
            'text':
            f"Predicción del consumo mensual(Litro/Mes) {actividad} en {section2}",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {
                'size': 30
            },
        },
                            legend={
                                "font": {
                                    "size": 18
                                },
                            },
                            xaxis={
                                "tickfont": {
                                    "size": 22
                                },
                                "title": {
                                    "text": "Meses",
                                    "font": {
                                        "size": 24
                                    },
                                },
                            },
                            yaxis={
                                "tickfont": {
                                    "size": 22
                                },
                                "title": {
                                    "text": "Consumo(Litro/Día)",
                                    "font": {
                                        "size": 24
                                    },
                                },
                            })
        final.add_trace(
            go.Scatter(x=df['ds'][35:35 + period + 1],
                       y=df['yhat1'][35:35 + period + 1],
                       mode='lines',
                       name='Prediction'))
        final.add_trace(
            go.Scatter(x=df.loc[:34, 'ds'],
                       y=df.loc[:34, 'y'],
                       mode='markers',
                       name='Actual',
                       line_color="firebrick"))

        fig_param = m.plot_parameters(plotting_backend="plotly")
        fig_param.update_layout(
            title={
                'text':
                f"Análisis de la tendencia del consumo {actividad} en {section2}",
                'y': 0.99,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {
                    'size': 16
                },
            })

    else:
        final = go.Figure()
        final.add_trace(
            go.Scatter(x=df['ds'],
                       y=df["yhat1"],
                       mode='lines',
                       name='Training set'))
        final.update_layout(title={
            'text':
            f"Predicción del consumo diario(Litro/Día) {actividad} en {section2}",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {
                'size': 30
            },
        },
                            legend={
                                "font": {
                                    "size": 18
                                },
                            },
                            xaxis={
                                "tickfont": {
                                    "size": 22
                                },
                                "title": {
                                    "text": "Meses",
                                    "font": {
                                        "size": 24
                                    },
                                },
                            },
                            yaxis={
                                "tickfont": {
                                    "size": 22
                                },
                                "title": {
                                    "text": "Consumo(Litro/Día)",
                                    "font": {
                                        "size": 24
                                    },
                                },
                            })
        final.add_trace(
            go.Scatter(x=df['ds'][1078:1078 + period + 1],
                       y=df['yhat1'][1078:1078 + period + 1],
                       mode='lines',
                       name='Prediction'))
        final.add_trace(
            go.Scatter(x=df['ds'],
                       y=df['y'],
                       mode='markers',
                       name='Actual',
                       line_color="firebrick"))

        fig_param = m.plot_parameters(plotting_backend="plotly")
        fig_param.update_layout(
            title={
                'text':
                f"Análisis de la tendencia del consumo {actividad} en {section2}",
                'y': 0.99,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {
                    'size': 16
                },
            })

    final.add_annotation(text=f"RMSE: {round(cost,2)}",
                         xref="paper",
                         yref="paper",
                         x=0.9,
                         y=0.1)

    if save_img == True:

        path = "./model/modelv2/images/"
        if mensual == True:
            path += "mensual/" + section2 + "/"
        else:
            path += "diario/" + section2 + "/"

        if not os.path.exists(path):
            os.makedirs(path)

        fig_param.write_image(path + f"param_{actividad}_{section2}.jpeg")
        final.write_image(path + f"prediction_{actividad}_{section2}.jpeg",
                          width=1980,
                          height=1080)

    if show == True:
        fig_param.show()
        final.show()

    if save_model == True:
        #####Saves the model's weights and state in the file model.pth, which allows the continuous update once it is deployed on the website.

        path = "./model/modelv2/save_model/"
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(m.model.state_dict(),
                   path + f"model_{actividad}_{section2}.pth")

    return m, pred_train, pred_next, final, fig_param, metrics  #pred_train -> train + test


data_location = [
    "./model/modelv2/sum_mensual_comercial.xlsx"
    "./model/modelv2/data/DOMESTICO_NORM_resum.xlsx",
    "./model/modelv2/data/INDUS_NORM_resum.xlsx"
    "./model/modelv2/sum_diario_comercial.xlsx"
]

zonas = [
    "BADALONA", "BARCELONA", "BEGUES", "CASTELLDEFELS", "CERDANYOLA",
    "CORNELLA", "EL PAPIOL", "ESPLUGUES", "GAVA", "L'HOSPITALET LLOBR.",
    "LA LLAGOSTA", "LES BOTIGUES SITGES", "MONTCADA I REIXAC", "MONTGAT",
    "PALLEJA", "SANT ADRIA", "SANT BOI", "SANT CLIMENT LLOB.", "SANT CUGAT",
    "SANT FELIU LL.", "SANT JOAN DESPI", "SANT JUST DESVERN",
    "STA.COLOMA CERVELLO", "STA.COLOMA GRAMENET", "TORRELLES LLOBREGAT",
    "VILADECANS"
]

if __name__ == "__main__":
    print("prediction model:\n")

    ###Noted that the last available data month(2021 December) is not complete, which shows a significant decay in the representation.
    data_location = "./model/modelv2/data/sum_mensual_comercial.xlsx"
    model = ARnet(data_location,
                  "BARCELONA",
                  save_prediction=True,
                  save_img=True,
                  save_model=True,
                  show=False)

    ###Noted that the last available data month(2021 December) is not complete, which shows a significant decay in the representation.
    # data_location = "./model/modelv2/data/sum_diario_comercial.xlsx"
    # model = ARnet(data_location,
    #               "BARCELONA",
    #               save_prediction=True,
    #               save_img=True,
    #               save_model=True,
    #               show=False)
