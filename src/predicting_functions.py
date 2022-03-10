import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import h2o
from h2o.automl import H2OAutoML

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, roc_curve, auc

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier


def prueba_modelo(modelo, X_train, y_train, X_test, y_test):
    '''Función para entrenar y testear modelos de forma rápida'''

    modelo.fit(X_train, y_train)

    train_score=modelo.score(X_train, y_train)  
    test_score=modelo.score(X_test, y_test)

    print(modelo)
    print('Train:', train_score)
    print('Test:', test_score) 
    print('\n')


def export_modelo(modelo, X_train, y_train, X_test):

    modelo.fit(X_train, y_train)

    sample=pd.read_csv('./Data/sample.csv')
    sample.price=modelo.predict(X_test)
    sample.to_csv('./Data/sample.csv',index=False)


def h2o_function(n_models, usecols ):
    h2o.init()

    train = pd.read_csv('./Data/train_clean.csv', usecols=usecols)
    test = pd.read_csv('./Data/test_clean.csv', usecols=usecols)
    train_price =pd.read_csv('./Data/train_clean.csv', usecols=["price"])

    train = pd.concat([train, train_price], axis=1)

    train_h2o = h2o.H2OFrame(train)
    test_h2o = h2o.H2OFrame(test)

    X=train_h2o.columns
    y='price'
    X.remove(y)

    train_h2o[y] = train_h2o[y]

    aml=H2OAutoML(max_models = (n_models))

    aml.train(x=X, y=y, training_frame=train_h2o)

    print(aml.leaderboard)

    sample=pd.read_csv('./Data/sample.csv')

    sample.price = aml.leader.predict(test_h2o).as_data_frame()

    sample.to_csv('./Data/sample.csv',index=False)



def roc_curve_plot(modelo, x_train, x_test, y_train, y_test):
    '''
    This function train a model and print the roc_curve
    
    Parameters:
    modelo (machine learning model): Modelo to be adjust and plot the roc curve

    x_train (array): Array to train the model 

    x_test (array): Array to test the model 

    y_train (array): Array to train the model (solution)

    y_test (array): Array to train the model (solution)
    '''
    modelo.fit(X=x_train, y=y_train)
    fpr, tpr, _ = roc_curve(y_test, modelo.predict_proba(x_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='red', label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="upper left")
    plt.show()


def function_prueba_datos(archivo):
    '''
    This function takes an array with a target column, undersample it, trains a lgbm model and print the roc_curve
    
    Parameters:
    archivo (array): Array to be undersampled and fit, predict with the lgbm model.
    '''
    train = pd.read_csv('/media/rodrigo/Rodrigo/'+archivo+'.csv')

    under = RandomUnderSampler(sampling_strategy=1)

    X, y = under.fit_resample(train.drop(columns=['target']), train.target)

    train = pd.concat([y, X], axis=1)

    X = train.drop(columns=['target'])
    y=train.target


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    lgbm = LGBMClassifier(n_jobs=-1)

    lgbm = lgbm.fit(X_train, y_train,eval_metric='auc',eval_set=(X_test , y_test),verbose=50,early_stopping_rounds= 50)



    fpr, tpr, _ = roc_curve(y_test, lgbm.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='red', label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="upper left")
    plt.show()