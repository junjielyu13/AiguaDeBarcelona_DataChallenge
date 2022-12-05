#pip install neuralprophet
import pandas as pd
import torch
from neuralprophet import NeuralProphet
import os
import plotly.graph_objects as go

import numpy as np


def ARnet_dia(data_location,
              section2,
              period=4,
              save_prediction=False,
              save_img=False,
              save_model=False,
              show=False):
    section1 = pd.read_excel(data_location)
    consumo = data_location.split("\\")[-1][:-3]

    def get_var_name(variable):
        globals_dict = globals()
        return [
            var_name for var_name in globals_dict
            if globals_dict[var_name] is variable
        ]

    if section2 in section1.columns[1:]:
        df = pd.DataFrame({"ds": section1.iloc[:, 0], "y": section1[section2]})
        m = NeuralProphet(loss_func=torch.nn.L1Loss)
        df_train, df_test = m.split_df(df, freq="D", valid_p=0.1)
        metrics = m.fit(df_train, freq="D", validation_df=df_test)
        metrics.tail(1)

        pred_train = m.predict(df)
        pred_test = m.predict(df_test)
        next_dataset = m.make_future_dataframe(df_test, periods=period)
        pred_next = m.predict(df=next_dataset)

        df = pd.concat([pred_train, pred_next])
        if save_prediction == True:
            df.to_csv(
                rf"C:\Users\23675\OneDrive\桌面\ABChallenge\Neuralprophet result\result_{consumo}_{section2}.csv"
            )
        final = go.Figure()
        final.add_trace(
            go.Scatter(x=df['ds'],
                       y=df["yhat1"],
                       mode='lines',
                       name='Training set'))
        final.update_layout(title={
            'text': f"Predicción del consumo diario(Litro/Día) en {section2}",
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
        final.add_trace(
            go.Scatter(x=df['ds'][971:1078],
                       y=df['yhat1'][971:1078],
                       mode='lines',
                       name='test set',
                       line_color="#32CD32"))
        fig_param = m.plot_parameters(plotting_backend="plotly")
        fig_param.update_layout(
            title={
                'text': f"Análisis de la tendencia del consumo en {section2}",
                'y': 1,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            })
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
        return m, pred_train, pred_next, final, fig_param  #pred_train -> train + test

    else:
        raise ValueError("No existe esta zona en nuestro dataset")


###Noted that the last available data month(2021 December) is not complete, which shows a significant decay in the representation.

## data_location = r"C:\Users\23675\OneDrive\桌面\ABChallenge\data\sum_diario_comercial.xlsx"
data_location = "./model/modelv2/sum_diario_comercial.xlsx"
model, pred_train, pred_next, fig_final, fig_param = ARnet_dia(data_location,
                                                               "BARCELONA",
                                                               show=True)
# model,pred_train,pred_next,fig_final,fig_param = ARnet_dia(data_location,"BADALONA",show =True)
# model,pred_train,pred_next,fig_final,fig_param = ARnet_dia(data_location,"GAVA",show = True)