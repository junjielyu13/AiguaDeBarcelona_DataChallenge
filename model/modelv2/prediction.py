#pip install neuralprophet
import pandas as pd
import torch
from neuralprophet import NeuralProphet
import os
import plotly.graph_objects as go

import numpy as np
#pip install neuralprophet
import pandas as pd
import torch
import kaleido
from neuralprophet import NeuralProphet
import os
import numpy as np


#pip install neuralprophet
import pandas as pd
import torch
import kaleido
from neuralprophet import NeuralProphet
import os
import numpy as np


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
    print(actividad)
    print(mensual)
    print(consumo)
    # def get_var_name(variable):
    #     globals_dict = globals()
    #     return [var_name for var_name in globals_dict if globals_dict[var_name] is variable]
    if mensual == True:
        df = pd.DataFrame({"ds": section1.iloc[:, 1], "y": section1[section2]})
        m = NeuralProphet(loss_func=torch.nn.L1Loss)
        df_train, df_test = m.split_df(df, freq="M", valid_p=0.1)
        metrics = m.fit(df_train, freq="M", validation_df=df_test)
    else:
        df = pd.DataFrame({"ds": section1["FECHA"], "y": section1[section2]})
        m = NeuralProphet(loss_func=torch.nn.L1Loss)
        print("--------------------------------")
        df_train, df_test = m.split_df(df, freq="D", valid_p=0.1)
        metrics = m.fit(df_train, freq="D", validation_df=df_test)
    cost = metrics.loc[metrics.shape[0] - 1, "RMSE"]
    pred_train = m.predict(df)
    next_dataset = m.make_future_dataframe(df_test, periods=period)
    pred_next = m.predict(df=next_dataset)

    df = pd.concat([pred_train, pred_next])

    if save_prediction == True:
        df.to_csv(
            rf"C:\Users\23675\OneDrive\桌面\ABChallenge\Neuralprophet result\result_{consumo}_{section2}.csv"
        )

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
                'y': 1,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
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
            'yanchor': 'top'
        },
                            xaxis_title="Meses",
                            yaxis_title="Consumo(Litro/Día)")

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
                'y': 1,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            })
    final.add_annotation(text=f"RMSE: {round(cost,2)}",
                         xref="paper",
                         yref="paper",
                         x=0.9,
                         y=0.1)
    if save_model == True:
        torch.save(
            m.model.state_dict(),
            rf"C:\Users\23675\OneDrive\桌面\ABChallenge\Neuralprophet result\model_{consumo}_{section2}.pth"
        )
    if save_img == True:
        final.write_image(
            rf"C:\Users\23675\OneDrive\桌面\ABChallenge\Neuralprophet result\result_{consumo}_{section2}.jpeg"
        )
    if show == True:
        fig_param.show()
        final.show()
    #1
    if save_model == True:

        torch.save(
            m.model.state_dict(),
            rf"C:\Users\23675\OneDrive\桌面\ABChallenge\Neuralprophet result\model_{consumo}_{section2}.pth"
        )  #####Saves the model's weights and state in the file model.pth, which allows the continuous update once it is deployed on the website.
    return m, pred_train, pred_next, final, fig_param, metrics  #pred_train -> train + test


###Noted that the last available data month(2021 December) is not complete, which shows a significant decay in the representation.
data_location = "./model/modelv2/sum_mensual_comercial.xlsx"
model, pred_train, pred_next, fig_final, fig_param, metrics = ARnet(
    data_location, "BARCELONA", show=True)
# model,pred_train,pred_next,fig_final,fig_param = ARnet_dia(data_location,"BADALONA",show =True)
# model,pred_train,pred_next,fig_final,fig_param = ARnet_dia(data_location,"GAVA",show = True)


###Noted that the last available data month(2021 December) is not complete, which shows a significant decay in the representation.

## data_location = r"C:\Users\23675\OneDrive\桌面\ABChallenge\data\sum_diario_comercial.xlsx"
data_location = "./model/modelv2/sum_diario_comercial.xlsx"
model, pred_train, pred_next, fig_final, fig_param = ARnet(data_location,
                                                           "BARCELONA",
                                                           show=True)
# model,pred_train,pred_next,fig_final,fig_param = ARnet_dia(data_location,"BADALONA",show =True)
# model,pred_train,pred_next,fig_final,fig_param = ARnet_dia(data_location,"GAVA",show = True)