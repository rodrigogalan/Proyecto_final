import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, roc_curve, auc

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier




def train_test(train):
    '''
    This function takes an array and divide it in a train part and in a test part

    Parameters:
    train (array): Array to be divided

    Returns:
    X_train (array): array with data in the train part
    X_test (array): array with data in the test part
    y_train (array): array with the solution in the train part
    y_test (array): array with the solution in the test part
    '''
    X = train.drop(columns=['target'])
    y=train.target
    return train_test_split(X, y, test_size=0.2)


def confusion_matrix_total(y_test, y_pred):
    '''
    This function takes two arrays, a predicction and the solution and returns a confusion matrix

    Parameters:
    y_test (array): array with the solution
    y_pred (array): array with the prediction
    '''
    ax=sns.heatmap(confusion_matrix(y_test, y_pred)/sum(sum(confusion_matrix(y_test, y_pred))), annot=True)

    plt.title('Matriz confusion')
    plt.ylabel('Verdad')
    plt.xlabel('Prediccion')
    plt.show();

def reg_model(modelo, x_train, x_test, y_train, y_test):
    '''
    This function trains a regresion model and returns the parameters needed to calculate the roc curve and the predictions
    
    Parameters:
    modelo (machine learning model): Modelo to be adjust and plot the roc curve
    x_train (array): Array to train the model 
    x_test (array): Array to test the model 
    y_train (array): Array to train the model (solution)
    y_test (array): Array to train the model (solution)

    Returns:
    fpr (array):   Increasing false positive rates 
    tpr (array): Increasing true positive rates
    roc_auc (float): area under the roc curve
    '''
    modelo.fit(X=x_train, y=y_train)
    y_pred=modelo.predict(X=X_test)

    fpr, tpr, _ = roc_curve(y_test, modelo.predict_proba(x_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc, y_pred


def roc_curve_plot(fpr, tpr, roc_auc):
    '''
    This function prints a roc curve
    
    Parameters:
    fpr (array):   Increasing false positive rates 
    tpr (array): Increasing true positive rates
    roc_auc (float): area under the roc curve
    '''

    plt.figure()
    plt.plot(fpr, tpr, color='red', label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="upper left")
    plt.show()


def export_modelo(y_pred):
    '''
    This funcition takes a solution array and exports it
    '''

    sample=pd.read_csv('./Data/sample.csv')
    sample.price=y_pred
    sample.to_csv('./Data/sample.csv',index=False)






























































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