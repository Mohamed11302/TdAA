from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

results = []
def calculo():
    #loading the data
    df = pd.read_csv('../train_data.csv')
    df_test = pd.read_csv('../test_data.csv')

    most_frequent = df['Garden'].mode()[0]
    df['Garden'].fillna(most_frequent, inplace=True)
    df_test['Garden'] = df_test['Garden'].fillna(value =0)

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

    df['Geo_Code'] = df['Geo_Code'].fillna(value = -1, inplace=True)
    df_test['Geo_Code'] = df_test['Geo_Code'].fillna(value = -1, inplace=True)

    df['NumberOfWindows'].value_counts()


    #change to numerical values
    df.Building_Painted.replace(('N','V'),(1,0), inplace = True)
    df_test.Building_Painted.replace(('N','V'),(1,0), inplace = True)
    df.Building_Fenced.replace(('N','V'),(1,0), inplace = True)
    df_test.Building_Fenced.replace(('N','V'),(1,0), inplace = True)
    df.Garden.replace(('V','O'),(1,0), inplace = True)
    df_test.Garden.replace(('V','O'),(1,0), inplace = True)
    df.Settlement.replace(('U','R'),(1,0), inplace = True)
    df_test.Settlement.replace(('U','R'),(1,0), inplace = True)

    df = df.drop("Geo_Code", axis = 1)
    df_test = df_test.drop("Geo_Code", axis = 1)

    features_selected = df.select_dtypes(include=['int64', 'float64']).columns.tolist()[:-1]

    # more cleaning to get x and y labels
    y = df.Claim
    X = df[features_selected]
    #splitting data into 30% test and 70% train
    X_train,X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)
    # Define el espacio de búsqueda de hiperparámetros para GaussianNB
    param_dist_random = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
        'priors': [None,[0.1, 0.9],[0.2, 0.8],[0.3, 0.7], [0.4, 0.6], [0.5, 0.5], [0.6, 0.4],[0.7, 0.3],[0.8, 0.2],[0.9, 0.1]],  # Ajusta los priors según tus necesidades
    }

    # Crea una instancia del modelo
    model = GaussianNB()

    # Realiza una búsqueda aleatoria utilizando la métrica adecuada (por ejemplo, 'roc_auc' si es una clasificación binaria)
    random_search = GridSearchCV(model, param_grid=param_dist_random, cv=3, scoring='roc_auc')

    # Ajusta la búsqueda aleatoria a tus datos
    random_search.fit(X_train, y_train)

    # Obtiene los mejores hiperparámetros encontrados
    best_params_random = random_search.best_params_

    # Luego puedes usar los mejores hiperparámetros para ajustar el modelo y hacer predicciones
    best_model = GaussianNB(**best_params_random)
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_pred)
    #print(f'ROC AUC Score: {roc_auc}')

    df_test["Claim"] = best_model.predict(df_test[features_selected])
    df_test[["Customer Id", "Claim"]].to_csv("../Submits/sub_gaussianNB_Opt.csv", index=False)
    return[roc_auc,(f"Mejores parámetros: {best_params_random}")]

if __name__ == '__main__':
    best=[0,""]
    while True:
        s=calculo()
        results.append(s)
        if s[0]>best[0]:
            print(s)
            best=s