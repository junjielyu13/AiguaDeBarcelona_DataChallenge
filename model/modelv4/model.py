###Noted PREDECIR
import pandas as pd
import torch
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet
import os
import plotly.figure_factory as ff
from neuralprophet import set_random_seed
import numpy as np
#from sklearn.preprocessing import StandardScaler

set_random_seed(0)


def NProphet(data_location,
             section2,
             period=4,
             save_prediction=True,
             save_img=True,
             save_model=True,
             pred_show=False,
             para_show=False,
             threshold_qt=0.95,
             anomaly_detect=True):

    pi = 0.8  # prediction interval
    qts = [(1 - pi) / 2,
           pi + (1 - pi) / 2]  # quantiles based on the prediction interval

    section1 = pd.read_excel(data_location)
    consumo = data_location.split("\\")[-1][:-3]
    mensual = consumo.split("_")[1] == "mensual"
    actividad = consumo.split("_")[2][:-2]

    m = NeuralProphet(loss_func=torch.nn.HuberLoss,
                      quantiles=qts,
                      weekly_seasonality=7,
                      growth="linear",
                      yearly_seasonality=True)  #prediction #weekly

    d = NeuralProphet(loss_func=torch.nn.HuberLoss,
                      quantiles=qts,
                      weekly_seasonality=7,
                      growth="linear",
                      yearly_seasonality=True)  #detection

    if mensual == True:
        df = pd.DataFrame({"ds": section1.iloc[:, 1], "y": section1[section2]})
        last_row = df.tail(1)
        df = df.drop(df.index[-1])
        metrics = m.fit(df, freq="M")
    else:
        df = pd.DataFrame({"ds": section1["FECHA"], "y": section1[section2]})
        metrics = m.fit(df, freq="D")  #progress ="plot-all"

    if anomaly_detect == True:
        if mensual == True:
            df_train, df_test = m.split_df(df, freq="M", valid_p=0.2)
        else:
            df_train, df_test = m.split_df(df, freq="D", valid_p=0.03)
        metrics_test = d.fit(df_train, freq="D",
                             validation_df=df_test)  #progress ="plot-all"
        det_train = d.predict(df_train)
        det_test = d.predict(df_test)
        det_final = pd.concat([df_train, det_test], axis=0)

    cost = metrics.loc[metrics.shape[0] - 1, "RMSE"]
    pred_train = m.predict(df)
    next_dataset = m.make_future_dataframe(df, periods=period)
    pred_next = m.predict(df=next_dataset)
    df_final = pd.concat([pred_train, pred_next], axis=0)  # for bounds

    ###detectar anomalias
    det_train["train_mae_loss"] = [
        np.mean(np.abs(det_train.loc[i, "y"] - det_train.loc[i, "yhat1"]))
        for i in range(len(det_train["y"]))
    ]
    # print(pred_train["train_mae_loss"])
    det_test["test_mae_loss"] = [
        np.mean(np.abs(det_test.loc[i, "y"] - det_test.loc[i, "yhat1"]))
        for i in range(len(det_test["y"]))
    ]

    final = go.Figure()
    fig_param = m.plot_parameters(plotting_backend="plotly")

    if mensual == True:
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

    if mensual == True:
        anomaly_detect = False

    if anomaly_detect == True:
        plt.hist(det_train["train_mae_loss"])
        plt.hist(det_test["test_mae_loss"])
        anomaly_sorted = det_train.sort_values(
            by='train_mae_loss')["train_mae_loss"]
        THRESHOLD = anomaly_sorted.quantile(threshold_qt)
        det_train["threshold"] = THRESHOLD
        det_test["threshold"] = THRESHOLD
        det_train["anomaly"] = det_train.train_mae_loss > det_train.threshold
        ####print anomaly value
        anomaly_values_train = det_train[
            det_train.anomaly ==
            True]  ## tell you which are the anomalies' values according to the threshold in the training set
        det_test["anomaly"] = det_test.test_mae_loss > det_test.threshold
        ####print anomaly value
        anomaly_values_test = det_test[
            det_test.anomaly ==
            True]  ## tell you which are the anomalies' values according to the threshold in the training set

        detect = go.Figure()
        detect.add_trace(
            go.Scatter(x=df_final['ds'],
                       y=df_final["yhat1 90.0%"],
                       name='upper bound',
                       line=dict(width=0),
                       marker=dict(color="#444"),
                       showlegend=False))
        detect.add_trace(
            go.Scatter(x=df_final['ds'],
                       y=df_final["yhat1 10.0%"],
                       name='lower bound',
                       line=dict(width=0),
                       fillcolor='rgba(68, 68, 68, 0.3)',
                       marker=dict(color="#444"),
                       fill='tonexty',
                       showlegend=False))
        detect.add_trace(
            go.Scatter(x=det_train['ds'],
                       y=det_train["yhat1"],
                       mode='lines+markers',
                       name='Training set',
                       marker=dict(color="#3366CC")))
        detect.add_trace(
            go.Scatter(x=det_test['ds'],
                       y=det_test['yhat1'],
                       mode='lines',
                       name='test set'))
        detect.add_trace(
            go.Scatter(x=df_final['ds'],
                       y=df_final['y'],
                       mode='lines',
                       name='Actual',
                       line_color="firebrick"))
        detect.update_layout(title={
            "text":
            f"Detección de errores {actividad} en {section2}",
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
        detect.add_scatter(x=det_train[det_train.anomaly == True]["ds"],
                           y=det_train[det_train.anomaly == True]["yhat1"],
                           mode="markers",
                           name=f'above the {threshold_qt} percentile',
                           marker=dict(color="rgb(166,216,84)"))
        detect.add_scatter(x=det_test[det_test.anomaly == True]["ds"],
                           y=det_test[det_test.anomaly == True]["yhat1"],
                           mode="markers",
                           marker=dict(color="rgb(255,217,47)"),
                           name='potential error detected')
        # detect.update_layout(template='plotly_dark')
        # anomaly.add_scatter(x=det_train.ds, y=det_train.threshold,name='threshold')
        # anomaly.add_scatter(x=det_test.ds, y=det_test[det_test.anomaly== True]["yhat1"],name='anomaly detected')

        detect.update_xaxes(
            type="date",
            range=["2021-09-01T00:00:00.000Z", "2022-1-01T00:00:00.000Z"])

    #Plot lines
    final.add_trace(
        go.Scatter(x=pred_train['ds'],
                   y=pred_train["yhat1"],
                   mode='lines+markers',
                   name='Training set'))
    final.add_trace(
        go.Scatter(x=pred_next['ds'],
                   y=pred_next['yhat1'],
                   mode='lines+markers',
                   name='predicción'))
    final.add_trace(
        go.Scatter(x=pred_train['ds'],
                   y=pred_train['y'],
                   mode='markers',
                   name='Actual',
                   line_color="firebrick"))
    final.add_annotation(text=f"RMSE: {round(cost,2)}",
                         xref="paper",
                         yref="paper",
                         x=0.9,
                         y=0.1)

    # lower upper bounds
    final.add_trace(
        go.Scatter(x=df_final['ds'],
                   y=df_final["yhat1 90.0%"],
                   name='upper bound',
                   line=dict(width=0),
                   marker=dict(color="#444"),
                   showlegend=False))
    final.add_trace(
        go.Scatter(x=df_final['ds'],
                   y=df_final["yhat1 10.0%"],
                   name='lower bound',
                   line=dict(width=0),
                   fillcolor='rgba(68, 68, 68, 0.3)',
                   marker=dict(color="#444"),
                   fill='tonexty',
                   showlegend=False))

    pred_train.to_csv(r"./df_final.csv")

    # Saves
    # if save_prediction == True:
    #     path = "./model/modelv4/prediction/"

    #     if mensual:
    #         path += "mensual/"
    #     elif anomaly_detect == True:
    #         path += "error/"
    #     else:
    #         path += "diario/"

    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     df.to_csv(path + f"result_{actividad}_{section2}.csv")

    # if save_model == True:
    #     path = "./model/modelv4/save_model/"

    #     if mensual == True:
    #         path += "mensual/"
    #     elif anomaly_detect == True:
    #         path += "error/"
    #     else:
    #         path += "diario/"

    #     if not os.path.exists(path):
    #         os.makedirs(path)

    #     torch.save(m.model.state_dict(),
    #                path + f"model_{actividad}_{section2}.pth")

    if save_img == True:

        path = "./model/modelv4/images/"

        if mensual == True:
            path += "mensual/"
        elif anomaly_detect == True:
            path += "error/"
        else:
            path += "diario/"

        if not os.path.exists(path):
            os.makedirs(path)

        # fig_param.write_image(path + f"param_{actividad}_{section2}.jpeg")
        # final.write_image(path + f"prediction_{actividad}_{section2}.jpeg",
        #                   width=1980,
        #                   height=1080)
        if anomaly_detect:
            detect.write_image(path + f"error_{actividad}_{section2}.jpeg",
                               width=1980,
                               height=1080)

    if pred_show == True:
        fig_param.show()
        final.show()
        if anomaly_detect:
            detect.show()

    return m, pred_train, pred_next, final, fig_param, metrics  #pred_train -> train + test


# mensual_Barcelona_comercial = r".\sum_mensual_comercial.xlsx"
# mensual_Barcelona_industrial = "./model/modelv2/data/sum_mensual_industrial.xlsx"
# mensual_Barcelona_domestic = r".\sum_mensual_domestic.xlsx"

# fig_param = {}

# ###Noted that the last available data month(2021 December) is not complete, which shows a significant decay in the representation.

# diario_Barcelona_comercial = "./model/modelv2/data/sum_diario_comercial.xlsx"

# diario_Barcelona_industrial = r"sum_diario_industrial.xlsx"
# diario_Barcelona_domestic = r"sum_diario_domestic.xlsx"

# param = NProphet(mensual_Barcelona_industrial, "BARCELONA", threshold_qt=0.95)

# param = NProphet(diario_Barcelona_comercial, "BARCELONA", threshold_qt=0.95)

data_set = [
    "./model/modelv2/data/sum_diario_comercial.xlsx",
    "./model/modelv2/data/sum_diario_domestic.xlsx",
    "./model/modelv2/data/sum_diario_industrial.xlsx",
    # "./model/modelv2/data/sum_mensual_comercial.xlsx",
    # "./model/modelv2/data/sum_mensual_domestic.xlsx",
    # "./model/modelv2/data/sum_mensual_industrial.xlsx"
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

for data in data_set:
    original = pd.read_excel(data)
    for zona in zonas:
        if zona in original.columns:
            model = NProphet(data, zona)
