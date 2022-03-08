from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def pca_transform(df, n_columns):
    '''
    This function recieves a dataframe and a number of columns and returns the dataframe after having applied the PCA process with the number of columns selected
    
    Parameters:
    df (DataFrame): DataFrame to applied PCA. It can not have objetct columns.
    n_columns (int): Number of columns of the dataframe after PCA process, it has to be equal of smaller than the total number of columns of the DataFrame

    Returns:
    DataFrame: DataFrame with the PCA process applied and the selected number of columns
    '''
    #normalised of the DataFrame
    df_final = StandardScaler().fit_transform(df)

    # Call PCA with the correct number of columns
    pca = PCA(n_components = n_columns)

    # Applied PCA
    return pca.fit(df_final)