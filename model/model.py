#pip install neuralprophet
import pandas as pd
import torch
from neuralprophet import NeuralProphet
import os
import numpy as np


def ARnet(section1, section2, save, split_size=0.1, period=4):
    if section2 in section1.columns[1:]:
        df = pd.DataFrame({"ds": section1.iloc[:, 1], "y": section1[section2]})
        m = NeuralProphet(loss_func=torch.nn.L1Loss)
        df_train, df_test = m.split_df(df, freq="M", valid_p=0.1)
        metrics = m.fit(df_train, freq="M", validation_df=df_test)
        metrics.tail(1)

        pred_train = m.predict(df)
        pred_test = m.predict(df_test)
        #prediction over the next period = "default:4" months
        next_dataset = m.make_future_dataframe(df_test, periods=4)
        pred_next = m.predict(df=next_dataset)
        print(pred_next.loc[:, "yhat1"])

        #plot
        fig_train = m.plot(pred_train,
                           plotting_backend="plotly",
                           ylabel="CONSUMO",
                           xlabel="Meses")

        fig_pred = m.plot(pred_next,
                          ylabel="predicción del CONSUMO",
                          plotting_backend="plotly")

        fig_param = m.plot_parameters(plotting_backend="plotly")
        #1

        fig_train.update_layout(title={
            'font': {
                'size': 22
            },
            'text': "Training - Test set",
            'y': 0.99,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
        },
                                legend={'font': {
                                    'size': 18
                                }},
                                margin={'autoexpand': True})
        fig_train.update_xaxes(title={"font": {"size": 18}})
        fig_train.update_yaxes(title={"font": {"size": 18}})

        fig_train.write_image("fig_train.png")

        #2
        fig_pred.update_layout(
            title={
                'text':
                f"Predicción del CONSUMO-{section2} en los siguientes 4 meses",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
            })

        fig_train.show()
        #fig_pred.show()
        #fig_param.show()

        return m, pred_train, pred_next, fig_train, fig_pred, fig_param  #pred_train -> train + test
    else:
        raise ValueError("No existe esta zona en nuestro dataset")


data_location = "./model/data/INDUS_NORM_resum.xlsx"
original = pd.read_excel(data_location)
original = original.iloc[1:, :]
model, pred_train, pred_next, fig_train, fig_pred, fig_param = ARnet(
    original, "BARCELONA", save="True")

### dataframe ###
##1. CONSUMO industrial
sections1 = [
    "INDUS_NORM_resum.xlsx", "DOMESTICO_NORM_resum.xlsx",
    "COMER_NORM_resum.xlsx"
]
data_location = os.path.join("./model/data", sections1[0])
sections2 = pd.read_excel(data_location).iloc[1:]


#for i, j be the element of both sections
#section1 = section1[i]
#section2 = section2[j]
def get_Dataset(section1, section2):
    data_location = os.path.join("./model/data", section1)  #改掉[0] ->[i]
    original = pd.read_excel(data_location)
    original = original.iloc[1:, :]
    ##Barcelona industrial
    zona = section2[0]  #改掉[0] -> [j]
    model, pred_train, pred_next, fig_train, fig_pred, fig_param = ARnet(
        section1, section2)
