import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix


from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from hyperopt import hp


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

def reg_model(modelo, X_train, X_test, y_train, y_test):
    '''
    This function trains a regresion model and returns the parameters needed to calculate the roc curve and the predictions
    
    Parameters:
    modelo (machine learning model): Modelo to be adjust and plot the roc curve
    X_train (array): Array to train the model 
    X_test (array): Array to test the model 
    y_train (array): Array to train the model (solution)
    y_test (array): Array to train the model (solution)

    Returns:
    fpr (array):   Increasing false positive rates 
    tpr (array): Increasing true positive rates
    roc_auc (float): area under the roc curve
    '''
    modelo.fit(X=X_train, y=y_train)
    y_pred=modelo.predict(X=X_test)

    fpr, tpr, _ = roc_curve(y_test, modelo.predict_proba(X_test)[:, 1])
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


def recuento_target(train):
    '''
    This function takes an array and returns the percentage of
    data with each target and draws a bar chart with the information.

    Parameters:
    train (array): Array to analyse
    '''

    train.target.value_counts().plot(kind='bar', figsize=(16,8));
    print('El porcentaje de 0 en el target es:', round(list(train.target.value_counts())[0]/train.shape[0]*100,2),"%")
    print('El porcentaje de 1 en el target es:', round(list(train.target.value_counts())[1]/train.shape[0]*100,2),"%")

def over_sampling(train):
    '''
    This function takes an array and returns another array over sampling

    Parameters:
    train (array): Array to over sample
    '''

    over = RandomOverSampler(sampling_strategy=1)
    X_over, y_over = over.fit_resample(train.drop(columns=['target']), train.target)
    train_over = pd.concat([y_over, X_over], axis=1)
    return train_over

def under_sampling(train):
    '''
    This function takes an array and returns another array under sampling

    Parameters:
    train (array): Array to under sample
    '''

    under = RandomUnderSampler(sampling_strategy=1)
    X_under, y_under = under.fit_resample(train.drop(columns=['target']), train.target)
    train_under = pd.concat([y_under, X_under], axis=1)
    return train_under


# Posibles parámetros para entrenar el modelo
params={'num_leaves' : hp.quniform('num_leaves', 1, 100, 2),
        'max_depth' : hp.quniform('max_depth', 1, 100, 2),
        'learning_rate' : hp.uniform('learning_rate', 0.0001, 1),
        'n_estimators': hp.quniform('n_estimators', 10, 1000, 25),
        'subsample':hp.uniform('subsample', 0.7, 1),
        'reg_lambda': hp.uniform('reg_lambda', 0, 1),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1)
        }
