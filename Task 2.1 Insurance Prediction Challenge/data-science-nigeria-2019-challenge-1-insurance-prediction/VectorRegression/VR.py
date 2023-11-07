import optuna
import numpy as np
from sklearn.svm import SVR
from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

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
# Divide tus datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Cargar tus datos de entrenamiento y prueba en DataFrames (reemplaza con tus datos reales)
# Asumiremos que tienes X_train, y_train, X_test y y_test

# Definir la función objetivo para la optimización de Optuna
def objective(trial):
    # Definir los hiperparámetros para la búsqueda
    C = trial.suggest_float('C', 1e-2, 1e2)
    epsilon = trial.suggest_float('epsilon', 1e-2, 1e2)
    kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
    degree = trial.suggest_int('degree', 2, 5)

    # Crear el modelo SVR con los hiperparámetros seleccionados
    model = SVR(C=C, epsilon=epsilon, kernel=kernel, degree=degree)

    # Entrenar el modelo con los datos de entrenamiento
    model.fit(X_train, y_train)

    # Realizar predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Calcular el valor de ROC AUC
    roc_auc = roc_auc_score(y_test, y_pred)

    return roc_auc

# Dividir tus datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un estudio Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50) 

# Obtener los mejores hiperparámetros encontrados
best_params = study.best_params

# Crear un modelo SVR con los mejores hiperparámetros
best_model = SVR(**best_params)

# Entrenar el modelo final con todos los datos de entrenamiento
best_model.fit(X, y)

# Realizar predicciones en tus datos de prueba o en nuevos datos
y_pred = best_model.predict(X_test)  # Cambia X_test por tus nuevos datos si es necesario

# Calcular el valor de ROC AUC en el conjunto de prueba o en los nuevos datos
roc_auc = roc_auc_score(y_test, y_pred)  # Cambia y_test por las etiquetas reales si es necesario

# Imprimir el valor de ROC AUC
print(f'Mejor ROC AUC: {roc_auc}')
best_model.fit(X_train, y_train)
df_test["Claim"] = best_model.predict(df_test[features_selected])

# Guarda las predicciones en un archivo CSV
df_test[["Customer Id", "Claim"]].to_csv("../Submits/submit_test_Vector.csv", index=False)


with open("../Parametros/parameters.txt","a") as f:
  f.write("VectorRegressor optuna: "+str(best_params)+" Roc_Aux: "+str(roc_auc)+"\n")
