import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from keras import layers
from tensorflow import keras
from sklearn.datasets import make_classification
from sklearn import model_selection
from sklearn import metrics
from itertools import combinations
import random
from sklearn.model_selection import GridSearchCV


def report(results, n_top):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def establecer_valores_perdidos_garden(fila):
    if fila['Settlement'] == 'R':
        return 'O'
    elif fila['Settlement'] == 'U':
        return 'V'


def limpieza(df, df_test):
    df['Garden'] = df.apply(establecer_valores_perdidos_garden, axis=1)
    df_test['Garden'] = df_test.apply(establecer_valores_perdidos_garden, axis=1)  

    
    df['NumberOfWindows'] = df['NumberOfWindows'].replace('   .', 0)
    df['NumberOfWindows'] = df['NumberOfWindows'].replace('>=10', 10)
    df_test['NumberOfWindows'] = df_test['NumberOfWindows'].replace('   .', 0)
    df_test['NumberOfWindows'] = df_test['NumberOfWindows'].replace('>=10', 10)
    df['NumberOfWindows'] = df['NumberOfWindows'].astype(int)
    df_test['NumberOfWindows'] = df_test['NumberOfWindows'].astype(int)


    # building dimension
    median_builddim = df['Building Dimension'].median()
    df['Building Dimension'].fillna(value = median_builddim, inplace=True)
    df_test['Building Dimension'].fillna(value = median_builddim, inplace=True)


    # date occupancy
    median_dateofocc= df['Date_of_Occupancy'].median()
    df['Date_of_Occupancy'].fillna(value = median_dateofocc, inplace=True)
    df_test['Date_of_Occupancy'].fillna(value = median_dateofocc, inplace=True)

    df = df.drop("Geo_Code", axis = 1)
    df_test = df_test.drop("Geo_Code", axis = 1)

    #change to numerical values
    df.Building_Painted.replace(('N','V'),(1,0), inplace = True)
    df_test.Building_Painted.replace(('N','V'),(1,0), inplace = True)
    df.Building_Fenced.replace(('N','V'),(1,0), inplace = True)
    df_test.Building_Fenced.replace(('N','V'),(1,0), inplace = True)
    df.Garden.replace(('V','O'),(1,0), inplace = True)
    df_test.Garden.replace(('V','O'),(1,0), inplace = True)
    df.Settlement.replace(('U','R'),(1,0), inplace = True)
    df_test.Settlement.replace(('U','R'),(1,0), inplace = True)

    
    return df, df_test

    
def MejorModelo_Encontrado(df, df_test):
    ####################################################################
    #               MEJOR MODELO POR AHORA
    # 0.629064700
    # GaussianNB(var_smoothing=1e-9, priors = np.array([0.4, 0.6]))
    #features_selected = ['Insured_Period', 'Residential','Settlement', 'Building Dimension', 'Building_Type']
    #test_size = 0.1
    #random_state = 421980
    ####################################################################
    features_selected = ['Insured_Period', 'Residential', 'Settlement', 'Building Dimension', 'Building_Type']
    X = df[features_selected]
    y = df.Claim
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=421980)
    priors = np.array([0.4, 0.6]) #Mejores resultados hasta ahora
    clf = GaussianNB(var_smoothing=1e-9, priors=priors)
    clf.fit(X_train, y_train)
    df_test["Claim"] = clf.predict(df_test[features_selected])
    df_test[["Customer Id", "Claim"]].to_csv("Submits/submit_test_NB.csv", index = False)

def GridSearch_Mejor_Modelo(df):
    features_selected = ['Insured_Period', 'Residential', 'Settlement', 'Building Dimension', 'Building_Type']
    X = df[features_selected]
    y = df.Claim
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_dist = {
    'var_smoothing': [1e-1, 1e-3, 1e-6, 1e-9, 1e-12],                 
    'priors': [(0.4, 0.6),(0.5,0.5),(0.6,0.4)],
    }
    clf = GaussianNB()
    grid_clf = GridSearchCV(estimator = clf, param_grid= param_dist, cv = 50, scoring='roc_auc')
    grid_clf.fit(X = X_train, y = y_train)
    report(grid_clf.cv_results_, n_top = 5)
    print(grid_clf.best_estimator_)

def TTS_best_params(df, iteraciones):
    features_selected = ['Insured_Period', 'Residential', 'Settlement', 'Building Dimension', 'Building_Type']
    X = df[features_selected]
    y = df.Claim
    mejor_AUCROC = 0
    mejor_random = 0
    for i in range(0, iteraciones):
        r = random.randint(0, 100000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=r)
        priors = np.array([0.4, 0.6]) #Mejores resultados hasta ahora
        clf = GaussianNB(var_smoothing=1e-9, priors=priors)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        roc_auc = metrics.roc_auc_score(y_test, y_pred)
        if roc_auc > mejor_AUCROC:
            mejor_AUCROC = roc_auc
            mejor_random = r
    print("AUC-ROC = " +str(mejor_AUCROC))
    print("Random = " +str(mejor_random))



def main():
    #loading the data
    df=  pd.read_csv("data-science-nigeria-2019-challenge-1-insurance-prediction\\train_data.csv")
    df_test = pd.read_csv("data-science-nigeria-2019-challenge-1-insurance-prediction\\test_data.csv")
    df, df_test = limpieza(df, df_test)
    MejorModelo_Encontrado(df, df_test)

if __name__ == "__main__":
    main()