import streamlit as st
import pandas as pd 
import plotly.graph_objects as go
from sklearn.manifold import TSNE


@st.cache_data(show_spinner=True)
def run_tsne(x,y,dimension,learning_rate,n_iteration,perplexity):
    model = TSNE(n_components=dimension, random_state=0, learning_rate=learning_rate, n_iter=n_iteration,
                 perplexity=perplexity)
    Y_fold_3d = model.fit_transform(x)
    Y_fold_3d = pd.DataFrame(Y_fold_3d, columns=[f'dim {i + 1}' for i in range(dimension)])
    Y_fold_3d["label"] = y.astype(str)
    return  Y_fold_3d

@st.cache_data
def create_figure(dimension,Y_fold_3d):
    traces = []
    if dimension == 3:
        for index, group in Y_fold_3d.groupby('label'):
            scatter = go.Scatter3d(
                name=f'Class {index}',
                x=group['dim 1'],
                y=group['dim 2'],
                z=group['dim 3'],
                mode='markers',
                marker=dict(
                    size=4,
                    symbol='circle'
                ))
            traces.append(scatter)

        fig = go.Figure(traces, layout=go.Layout(height=500,
                                                 margin=dict(l=0, r=0, b=0, t=30),
                                                 uirevision='foo'))
    else:
        for index, group in Y_fold_3d.groupby('label'):
            scatter = go.Scatter(
                name=f'Class {index}',
                x=group['dim 1'],
                y=group['dim 2'],
                mode='markers',
                marker=dict(
                    size=4.5,
                    symbol='circle'
                ))
            traces.append(scatter)

        fig = go.Figure(traces, layout=go.Layout(height=500,
                                                 margin=dict(l=0, r=0, b=0, t=30),
                                                 uirevision='foo'))
   
    return fig
@st.cache_data
def create_standard_figure():


    fig = go.Figure([go.Scatter3d(x=[0],y=[0],z=[0])], layout=go.Layout(height=500,
                                                 margin=dict(l=0, r=0, b=0, t=30),
                                                 uirevision='foo'))
                                                
    fig.update_layout(scene=dict(
		xaxis=dict(type="linear"),
		yaxis=dict(type="linear"),
		zaxis=dict(type="linear"),
		annotations=[
		dict(
		    showarrow=False,
		    x=0,
		    y=0,
		    z=0,
		    text="T-SNE Viewer ",
		    xanchor="left",
		    xshift=10,
		    opacity=0.7,
		    font=dict(
		   color="black",
		   size=36
	    		)
		    
		   	)
		    
		    ]))
        
    return fig

