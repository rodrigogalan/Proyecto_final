# Santander customer transaction prediction

<div align=center>
    <img src ="./images/santander_logo.png" alt="Banco Santander logo">
</div>

Este proyecto trata de resolver una problemática propuesta por el banco Santander en [Kaggle](https://www.kaggle.com/c/santander-customer-transaction-prediction/overview/description) donde el objetivo es, a través de un dataset de columnas, encontrar un modelo que permita predecir si un cliente va a efectuar o no una transacción.

## Documentos
### Jupyter notebooks
* [1-Analysis.ipynb](https://github.com/rodrigogalan/Proyecto_final/blob/main/1-Analysis.ipynb): exploración de los dataframes de datos
* [2-Reduction.ipynb](https://github.com/rodrigogalan/Proyecto_final/blob/main/2-Reduction.ipynb): reducción de las dimensiones de los datos
* [3-Models.ipynb](https://github.com/rodrigogalan/Proyecto_final/blob/main/3-Models.ipynb): prueba de modelos

### Funciones auxiliares
* [exploring_functions.py](https://github.com/rodrigogalan/Proyecto_final/blob/main/src/chaging_functions.py): archivo con funciones para analizar un dataframe y reducir 
* [changing_functions.py](https://github.com/rodrigogalan/Proyecto_final/blob/main/src/chaging_functions.py): archivo con funciones para aplicar PCA a un dataframe
* [predicting_functions.py](https://github.com/rodrigogalan/Proyecto_final/blob/main/src/chaging_functions.py): archivo para agilizar la prueba de modelos predictivos

## Librerias
* [numpy](https://numpy.org/doc/1.22/)
* [pandas](https://pandas.pydata.org/pandas-docs/stable/) 
* [sys](https://docs.python.org/3/library/sys.html)
* [seaborn](https://seaborn.pydata.org/)
* [plotly](https://plotly.com/python/)
* [sklearn](https://www.kite.com/python/docs/sklearn)
* [umap](https://umap-learn.readthedocs.io/en/latest/)
* [scipy](https://docs.scipy.org/doc/scipy/)
* [catboos](https://catboost.ai/en/docs/)
* [lightgbm](https://lightgbm.readthedocs.io/en/latest/Python-API.html)
* [imblearn](https://scikit-learn.org/stable/)
* [h2o](https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/intro.html)