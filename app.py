
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from dataset import load_wine_dataset , load_digits_dataset
from utils import run_tsne,create_figure,create_standard_figure
##### Set page config
st.set_page_config(
    page_title="T-SNE Streamlit Viewer",
    page_icon="✒️")


##### Side bar config

st.sidebar.title('TSNE parameters')
#st.sidebar.header('Configuration')
dataset_name= st.sidebar.selectbox('Load dataset',['None','Digits','Wine'],index=0)

dimension = st.sidebar.selectbox('Choose the dimension of embedded space  ',
                                  [2,3],index=0,)
perplexity = st.sidebar.slider(label='Perplexity',min_value=1,max_value=100,step=1,value=3)
n_iteration = st.sidebar.slider(label='Number of iterations',min_value=1,max_value=1000,step=100,value=700)
learning_rate = st.sidebar.slider(label='Learning rate',min_value=10,max_value=1000,step=10,value=10)

##### Streamlit app

tab1,tab2 = st.tabs(['Description','Viewer'])

with tab1:
    st.title('T-SNE data visualization demo :sunglasses:')
    st.header('Digits dataset in 2D and 3D space representation')
    st.markdown("""
    ### Introduction:
    In this tutorial we will show you an example of one of the most famous algorithm of dimensionality reduction,
    T-distributed Stochastic Neighbor Embedding which we called **T-SNE**
    ### What is **T-SNE**: ?
    
    T-SNE (t-distributed Stochastic Neighbor Embedding) is a **non-linear dimensionality reduction** algorithm for data **exploration and visualizing** high-dimensional data. It works by finding pairs of points in **the high-dimensional space** that are close together and mapping them to pairs of points in **the low-dimensional space** that are also close together. However, t-SNE also takes into account the global structure of the data, so that points that are close together in the high-dimensional space are also **likely** to be **close together** in the low-dimensional space, even if they are not directly connected.
    
    ### Paper:
    The Original paper has been written by **Maaten, L. v. d., & Hinton, G. E.** in 2008\n
    :point_right: You can check the original [paper](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)""")
    if dataset_name =='Digits':
        st.markdown("""
	    ### Dataset:
	    The digits dataset is a dataset of **handwritten digits** from 0 to 9. It is a popular dataset for machine learning and computer vision tasks,    such as **digit recognition** and **image classification**. The dataset contains 1797 images, of which 1439 are for training and 360 are for testing. The images are grayscale and have a resolution of 8x8 pixels.\n
	    In this example we used the [**Digits dataset**](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
	    from Sklearn library
	    """)
        st.code(
		"""
		import numpy as np
		from sklearn.datasets import load_digits

		# Load the digits dataset
		digits = load_digits()
		# Print the shape of the data
		print('data shape:',digits.data.shape)
		# Print the target values
		print('target:',digits.target)
		"""
		)
        with st.expander('**Output**',expanded=True):
             st.write('data shape:(1797, 64)')
             st.write('target:[0 1 2 3 4 5 6 7 8 9]')
             
    elif dataset_name=='Wine':
        st.markdown("""
	    ### Dataset:
	    The wine dataset is a classic and very easy **multi-class classification dataset**. It contains 178 samples of wine belonging to 3 	    different classes: Class 0 (Cabernet Sauvignon), Class 1 (Pinot Noir), and Class 2 (Syrah). Each sample is described by 13 features, 		    which include chemical properties such as alcohol, malic acid, and ash.
	    In this example we used the [**Wine dataset**](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html) from Sklearn library
	    """)
        st.code(
		"""
		from sklearn.datasets import load_wine

		wine = load_wine()
		print("Feature shape:", wine.data.shape)
		print("Target shape:", wine.target.shape)
		"""
		)
        with st.expander('**Output**',expanded=True):
             st.write('Feature shape: (178, 13)')
             st.write('Target shape: (178,)')
    else:
       st.subheader(":point_left: Please select a dataset from side bar ")
        


###Viewer

if dataset_name=='Digits':
   Y_fold,y_fold=load_digits_dataset()
   Y_fold_3d=run_tsne(x=Y_fold,y=y_fold,dimension=dimension,learning_rate=learning_rate,n_iteration=n_iteration,perplexity=perplexity)
   fig = create_figure(dimension=dimension,Y_fold_3d=Y_fold_3d)
elif dataset_name=='Wine':
   Y_fold,y_fold=load_wine_dataset()
   Y_fold_3d=run_tsne(x=Y_fold,y=y_fold,dimension=dimension,learning_rate=learning_rate,n_iteration=n_iteration,perplexity=perplexity)
   fig = create_figure(dimension=dimension,Y_fold_3d=Y_fold_3d)
else:
   fig=create_standard_figure()
   
with tab2:
    if dataset_name=='None':
       st.subheader("Welcome to this tutorial :clap:")
       st.subheader(":point_left: Please load a dataset from side bar")
       
    else:
       st.subheader(f'{dataset_name} has been :green[loaded] :tada: ')
       st.success('Successfully loaded ')
    st.plotly_chart(fig,theme='streamlit')
