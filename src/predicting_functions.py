import pandas as pd
import h2o
from h2o.automl import H2OAutoML


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
