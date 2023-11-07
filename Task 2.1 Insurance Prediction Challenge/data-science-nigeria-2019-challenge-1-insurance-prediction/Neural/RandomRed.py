from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, make_scorer
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_random_state

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


def create_model(layers, activation, optimizer):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i == 0:
            model.add(Dense(nodes, input_dim=X_train.shape[1]))
            model.add(activation)
        else:
            model.add(Dense(nodes))
            model.add(activation)
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Define los hiperparámetros y sus posibles valores
param_dist_random = {
    'layers': [(64,), (128, 64), (64, 32, 16)],
    'activation': ['relu', 'tanh', 'sigmoid'],
    'optimizer': ['adam', 'sgd', 'rmsprop'],
    'batch_size': [16, 32, 64, 128],
    'epochs': [10, 20, 30],
}

# Crea una instancia del modelo
model = keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, verbose=0)

# Realiza una búsqueda aleatoria utilizando la métrica adecuada (por ejemplo, 'roc_auc' si es una clasificación binaria)
random_search = RandomizedSearchCV(model, param_distributions=param_dist_random, n_iter=10, cv=3, scoring='roc_auc', random_state=42)

# Ajusta la búsqueda aleatoria a tus datos
random_search.fit(X_train, y_train)

# Obtiene los mejores hiperparámetros encontrados
best_params_random = random_search.best_params_

# Crea el modelo con los mejores hiperparámetros y ajústalo a tus datos
best_model = create_model(**best_params_random)
best_model.fit(X_train, y_train, batch_size=best_params_random['batch_size'], epochs=best_params_random['epochs'], verbose=0)

# Calcula el ROC AUC en el conjunto de prueba
y_pred = best_model.predict(X_test)
roc_auc = roc_auc_score(y_test, y_pred)
print(f'ROC AUC Score: {roc_auc}')
# Obtiene los mejores hiperparámetros
best_params = random_search.best_params_
s=(f"Mejores hiperparámetros: {best_params}")
s=s.replace('\'','')
s=s.replace(':','=')
print(s)
# Entrena el modelo final con los mejores hiperparámetros
best_model = random_search.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

roc_auc = roc_auc_score(y_test, y_pred)
print(f'ROC AUC Score: {roc_auc}')
df_test["Claim"] = best_model.predict(df_test[features_selected])
df_test[["Customer Id", "Claim"]].to_csv("../Submits/submit_test_rf_Rand.csv", index=False)
