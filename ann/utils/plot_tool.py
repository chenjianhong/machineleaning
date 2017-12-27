#coding:utf-8
import plotly.graph_objs as go
import plotly

def origin_marker_predict_line_plot(x,y,y_pred):
    trace1 = go.Scatter(
        x=x,
        y=y,
        mode='markers',
        name='markers1'
    )
    trace2 = go.Scatter(
        x=x,
        y=y_pred,
        mode='lines',
        name='lines2'
    )
    data = [trace1, trace2]
    # plotly.plotly.iplot(data, filename='tensor_demo')
    plotly.offline.plot(data, filename='tensor_demo.html')