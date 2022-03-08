import numpy as np
import pandas as pd

import pylab as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def pca_visualize(df):
    '''
    This function recieves a dataframe plot the evolution of the variance in function of the components in the PCA process
    
    Parameters:
    df (DataFrame): DataFrame to test variance evolution

    '''
    #normalised of the DataFrame
    df_temp = StandardScaler().fit_transform(df)

    # Call PCA with the correct number of columns
    pca = PCA()

    # Applied PCA
    pca.fit(df_temp)

    # Design plot
    plt.figure(figsize=(10, 5))

    plt.plot(np.cumsum(pca.explained_variance_ratio_))

    plt.xlabel('Numero de componentes')
    plt.ylabel('% varianza')
    plt.ylim([0, 1.01]);



def pca_transform(df1, df2, n_columns):
    '''
    This function recieves a dataframe and a number of columns and returns the dataframe after having applied the PCA process with the number of columns selected
    
    Parameters:
    df1 (DataFrame): DataFrame to fit and applied PCA. It can not have objetct columns.
    df2 (DataFrame): DataFrame to transform with the PCA fitted in the previous DataFrame.
    n_columns (int): Number of columns of the dataframe after PCA process, it has to be equal of smaller than the total number of columns of the DataFrame

    Returns:
    DataFrame: DataFrame with the PCA process applied and the selected number of columns
    '''
    #normalised of the DataFrame
    df_final1 = StandardScaler().fit_transform(df1)
    df_final2 = StandardScaler().fit_transform(df2)


    # Call PCA with the correct number of columns
    pca = PCA(n_components = n_columns)

    # Applied PCA
    return pd.DataFrame(pca.fit_transform(df_final1)), pd.DataFrame(pca.transform(df_final2))
