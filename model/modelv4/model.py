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
from sklearn.preprocessing import StandardScaler
set_random_seed(0)
def NProphet(data_location,section2,period=4,save_prediction=False,save_img = False,save_model = False,pred_show=False,para_show=False,threshold_qt =0.95,anomaly_detect=True):
    
    pi = 0.8  # prediction interval
    qts = [(1 - pi) / 2, pi + (1 - pi) / 2]  # quantiles based on the prediction interval

    section1 = pd.read_excel(data_location)
    consumo = data_location.split("\\")[-1][:-3]
    mensual = consumo.split("_")[1] == "mensual"
    actividad = consumo.split("_")[2][:-2]
    print(actividad)
    print(mensual)
    print(consumo)
    m = NeuralProphet(loss_func=torch.nn.HuberLoss,quantiles=qts,weekly_seasonality=7,growth= "linear",yearly_seasonality=True) #prediction #weekly
    d = NeuralProphet(loss_func=torch.nn.HuberLoss,quantiles=qts,weekly_seasonality=7,growth="linear",yearly_seasonality=True) #detection

    if mensual == True:
        df = pd.DataFrame({"ds":section1.iloc[:,1],"y":section1[section2]})
        last_row = df.tail(1)
        df = df.drop(df.index[-1])
        metrics = m.fit(df, freq="M")
    else:
        df = pd.DataFrame({"ds":section1["FECHA"],"y":section1[section2]})
        print("--------------------------------")
        metrics = m.fit(df, freq="D")#progress ="plot-all"

    if anomaly_detect == True:
        if mensual == True:
            print("--------------------------------")
            df_train, df_test = m.split_df(df, freq="M", valid_p=0.2)
        else:
            df_train, df_test = m.split_df(df, freq="D", valid_p=0.03)
        metrics_test = d.fit(df_train, freq="D",validation_df = df_test)#progress ="plot-all"
        det_train =d.predict(df_train)
        det_test = d.predict(df_test)
        det_final = pd.concat([df_train,det_test],axis = 0)






    cost=metrics.loc[metrics.shape[0]-1,"RMSE"]
    pred_train = m.predict(df)
    next_dataset = m.make_future_dataframe(df,periods=period)
    pred_next = m.predict(df=next_dataset)
    df_final = pd.concat([pred_train,pred_next],axis = 0) # for bounds


###detectar anomalias
    det_train["train_mae_loss"] = [np.mean(np.abs(det_train.loc[i, "y"] - det_train.loc[i, "yhat1"])) for i in range(len(det_train["y"]))]
    # print(pred_train["train_mae_loss"])
    det_test["test_mae_loss"] = [np.mean(np.abs(det_test.loc[i, "y"] - det_test.loc[i, "yhat1"])) for i in range(len(det_test["y"]))]



    final = go.Figure()
    fig_param = m.plot_parameters(plotting_backend="plotly")

    if mensual ==True:
        final.update_layout(
        title={
        'text': f"Predicción del consumo mensual(Litro/Mes) {actividad} en {section2}",'y':0.95,'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
        },
        xaxis_title="Meses",yaxis_title="Consumo(Litro/Mes)")
        fig_param.update_layout(
        title={
        'text': f"Análisis de la tendencia del consumo {actividad} en {section2}",'y':1,'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}
        )

    else:           
        final = go.Figure()
        final.update_layout(
        title={
        'text': f"Predicción del consumo diario(Litro/Día) {actividad} en {section2}",'y':0.95,'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
        },xaxis_title="Meses",yaxis_title="Consumo(Litro/Día)")

        fig_param.update_layout(    
        title={
        'text': f"Análisis de la tendencia del consumo {actividad} en {section2}",'y':1,'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}
        )
#         hist = go.Histogram(x=pred_train["ds"], y=pred_train["train_mae_loss"])
#         # kernel density estimation
#         # fig = ff.create_distplot(pred_train["ds"],pred_train["train_mae_loss"], curve_type='kde')

        #apply this threshold of difference in mae to the test set annd find out the anomaly in the test set.
        # det_test["anomaly"] =  det_test.train_mae_loss > .threshold
    if mensual == True:
        anomaly_detect =False
    if anomaly_detect == True:
        plt.hist(det_train["train_mae_loss"])
        plt.hist(det_test["test_mae_loss"])
        anomaly_sorted = det_train.sort_values(by='train_mae_loss')["train_mae_loss"]
        THRESHOLD = anomaly_sorted.quantile(threshold_qt)
        det_train["threshold"] = THRESHOLD
        det_test["threshold"] = THRESHOLD
        det_train["anomaly"] = det_train.train_mae_loss > det_train.threshold
        ####print anomaly value 
        anomaly_values_train = det_train[det_train.anomaly == True]  ## tell you which are the anomalies' values according to the threshold in the training set
        det_test["anomaly"] = det_test.test_mae_loss > det_test.threshold
        ####print anomaly value 
        anomaly_values_test = det_test[det_test.anomaly == True]  ## tell you which are the anomalies' values according to the threshold in the training set




        detect = go.Figure()
        detect.add_trace(go.Scatter(x=df_final['ds'], y=df_final["yhat1 90.0%"], name='upper bound',line=dict(width=0),marker=dict(color="#444"),showlegend=False))
        detect.add_trace(go.Scatter(x=df_final['ds'], y=df_final["yhat1 10.0%"], name='lower bound',line=dict(width=0),fillcolor='rgb(102, 102, 102)',marker=dict(color="#444"), fill='tonexty', showlegend=False))
        detect.add_trace(go.Scatter(x=det_train['ds'], y=det_train["yhat1"], mode='lines+markers', name='Training set',marker=dict(color="#3366CC"))) 
        detect.add_trace(go.Scatter(x=det_test['ds'], y=det_test['yhat1'],mode='lines',name='test set'))
        detect.add_trace(go.Scatter(x=df_final['ds'], y=df_final['y'],mode='lines',name='Actual',line_color="firebrick"))
        detect.update_layout(title={"text":"Detección de errores"})
        detect.add_scatter(x=det_train[det_train.anomaly ==True]["ds"], y=det_train[det_train.anomaly ==True]["yhat1"],mode="markers", name=f'above the {threshold_qt} percentile',marker=dict(color="rgb(166,216,84)"))
        detect.add_scatter(x=det_test[det_test.anomaly ==True]["ds"], y=det_test[det_test.anomaly ==True]["yhat1"],mode="markers",marker=dict(color="rgb(255,217,47)"), name='potential error detected')
        detect.update_layout(template='plotly_dark')
        # anomaly.add_scatter(x=det_train.ds, y=det_train.threshold,name='threshold')
        # anomaly.add_scatter(x=det_test.ds, y=det_test[det_test.anomaly== True]["yhat1"],name='anomaly detected')


        detect.update_xaxes(type="date", range=["2021-09-01T00:00:00.000Z", "2022-1-01T00:00:00.000Z"])
        detect.show()




    #Plot lines
    final.add_trace(go.Scatter(x=pred_train['ds'], y=pred_train["yhat1"], mode='lines+markers', name='Training set'))        
    final.add_trace(go.Scatter(x=pred_next['ds'], y=pred_next['yhat1'],mode='lines+markers',name='predicción'))
    final.add_trace(go.Scatter(x=pred_train['ds'], y=pred_train['y'],mode='markers',name='Actual',line_color="firebrick"))
    final.add_annotation(text=f"RMSE: {round(cost,2)}",
            xref="paper", yref="paper",
            x=0.9, y=0.1)


    # lower upper bounds
    final.add_trace(go.Scatter(x=df_final['ds'], y=df_final["yhat1 90.0%"], name='upper bound',line=dict(width=0),marker=dict(color="#444"),showlegend=False))
    final.add_trace(go.Scatter(x=df_final['ds'], y=df_final["yhat1 10.0%"], name='lower bound',line=dict(width=0),fillcolor='rgba(68, 68, 68, 0.3)',marker=dict(color="#444"), fill='tonexty', showlegend=False))

    #TODO
    # Create an animation object

    # frames = [go.Frame(data=[final])]
    # animation = go.Animation(frames=frames)

    # # Create a figure object and add the animation to it
    # fig = go.Figure(data=[detect], layout=go.Layout(updatemenus=[dict(type='buttons', buttons=[dict(label='Play', method='animate', args=[None])])]))
    # fig.update_layout(updatemode='linear')
    # fig.show()    

    # final.show()
    pred_train.to_csv(r"./df_final.csv")
    

    # Saves
    if save_prediction == True:
        df.to_csv(rf"C:\Users\23675\OneDrive\桌面\ABChallenge\Neuralprophet result\result_{consumo}_{section2}.csv")
    if save_model == True:
        torch.save(m.model.state_dict(),rf"C:\Users\23675\OneDrive\桌面\ABChallenge\Neuralprophet result\model_{consumo}_{section2}.pth")
    if save_img == True:
        final.write_image(rf"C:\Users\23675\OneDrive\桌面\ABChallenge\Neuralprophet result\result_{consumo}_{section2}.jpeg")
    if para_show == True:
        fig_param.show()
    if pred_show == True:
        final.show()
    #1
    if save_model == True:
        torch.save(m.model.state_dict(),rf"C:\Users\23675\OneDrive\桌面\ABChallenge\Neuralprophet result\model_{consumo}_{section2}.pth") #####Saves the model's weights and state in the file model.pth, which allows the continuous update once it is deployed on the website.
    return m,pred_train,pred_next,final,fig_param,metrics #pred_train -> train + test



mensual_Barcelona_comercial = r".\sum_mensual_comercial.xlsx"

mensual_Barcelona_industrial = r".\sum_mensual_industrial.xlsx"
mensual_Barcelona_domestic = r".\sum_mensual_domestic.xlsx"


fig_param = {}







###Noted that the last available data month(2021 December) is not complete, which shows a significant decay in the representation.

diario_Barcelona_comercial = r"sum_diario_comercial.xlsx"

diario_Barcelona_industrial = r"sum_diario_industrial.xlsx"
diario_Barcelona_domestic = r"sum_diario_domestic.xlsx"

model,pred_train,pred_next,fig_final,fig_param["BARCELONA"],metrics = NProphet(mensual_Barcelona_industrial,"BARCELONA",pred_show = True,para_show = True,threshold_qt=0.95)
model,pred_train,pred_next,fig_final,fig_param["BARCELONA"],metrics = NProphet(diario_Barcelona_comercial,"BARCELONA",pred_show = True,para_show = True,threshold_qt=0.95)  