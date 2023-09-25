from sklearn import datasets
import pandas as pd 
import numpy as np
import streamlit as st 


@st.cache_data(show_spinner=True)
def load_wine_dataset():
   x,y = datasets.load_wine(return_X_y=True, as_frame=True)
   x =x.values
   y=y.values
   return x,y


@st.cache_data(show_spinner=True)
def load_digits_dataset():
    digits = datasets.load_digits()
    X_fold,y_fold = np.reshape(digits["data"], (-1,8,8)), digits["target"]

    classes, counts_per_class = np.unique(y_fold, return_counts=True)
    K = len(classes)

    n_bits_classes = int(np.log2(K+1)) + 1 * (K + 1 - 2 ** int(np.log2(K+1)) > 0)
    classes_repr = {c: np.array(list(map(int, list(np.binary_repr(i+1, width=n_bits_classes)))))
                    for i, c in enumerate(classes)}
    # construct in-class id mapping
    ## maximum number of samples per class in the fold
    max_id = np.max(np.unique(y_fold, return_counts=True)[1])
    ## compute the minimum n_bits allowing to uniquely represent the ids if we start counting from 1
    n_bits_id = int(np.log2(max_id+1)) + 1 * (max_id +1 - 2 ** int(np.log2(max_id+1)) > 0)
    ## store the binary encoding for each possible id
    Y_codebook = np.array([list(map(int, list(np.binary_repr(i+1, width=n_bits_id)))) for i in range(max_id)])
    ## normalize the encoding to make the maximum intra-class distance lower than the minimum inter-class distance
    Y_codebook = Y_codebook /np.sqrt(max_id) # maybe sqrt(max_id) is a better choice

    # construct the new labels [[ class embedding, in-class id embedding], ... ]
    Y_fold = np.zeros((len(y_fold), n_bits_classes + n_bits_id))
    ## store y_fold in a dataframe, a pretext to use groupby later
    y_frame = pd.DataFrame({"y": y_fold})
    for a, group in y_frame.groupby("y"):
        ### classes encoding
        A = np.repeat(classes_repr[a][None, :], len(group), axis=0)
        ### id encoding
        B = Y_codebook[:len(group)]
        ### Labels of the fold
        C = np.hstack((A,B))
        ### store the labels at the right positions
        Y_fold[group.index , :] = C
    return Y_fold,y_fold
