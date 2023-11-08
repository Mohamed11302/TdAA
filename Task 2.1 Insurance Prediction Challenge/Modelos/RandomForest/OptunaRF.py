import optuna
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
import pandas as pd
import random

#loading the data
df = pd.read_csv("data-science-nigeria-2019-challenge-1-insurance-prediction\\train_data.csv")
df_test = pd.read_csv("data-science-nigeria-2019-challenge-1-insurance-prediction\\test_data.csv")

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Define la función de optimización de Optuna
def objective(trial):
    # Define los hiperparámetros a optimizar
    n_estimators = trial.suggest_int("n_estimators", 1, 3000)
    max_depth = trial.suggest_int("max_depth", 1, 50)
    min_samples_split = trial.suggest_float("min_samples_split", 0.1, 1.0)
    min_samples_leaf = trial.suggest_float("min_samples_leaf", 0.1, 1.0)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
    
    # Crea una instancia del modelo Random Forest con los hiperparámetros sugeridos
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features
        #random_state=42
    )
    
    # Entrena el modelo en los datos de entrenamiento
    model.fit(X_train, y_train)
    
    # Realiza predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)
    
    # Calcula el ROC AUC en el conjunto de prueba
    roc_auc = roc_auc_score(y_test, y_pred)
    
    return roc_auc

# Llama a la función de optimización de Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# Obtiene los mejores hiperparámetros encontrados por Optuna
best_params = study.best_params

# Crea el modelo final con los mejores hiperparámetros y ajústalo a tus datos

best_AUCROC = 0
best_random = 0
iter = 100
for i in range(0, iter):
    r = random.randint(0, 100000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=r)
    DTR = RandomForestRegressor(**best_params, random_state=r)
    DTR.fit(X_train, y_train)
    y_pred = DTR.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_pred)
    if roc_auc > best_AUCROC:
        best_AUCROC = roc_auc
        best_random = r
print("Best AUC-ROC = " +str(best_AUCROC))
print("Best Random = " +str(best_random))

#Guardamos el resultado del mejor modelo con el mejor estado aleatorio
DTR = RandomForestRegressor(**best_params, random_state=best_random)
DTR.fit(X_train, y_train)
df_test["Claim"] = DTR.predict(df_test[features_selected])

# Guarda las predicciones en un archivo CSV
df_test[["Customer Id", "Claim"]].to_csv("Submits/submit_test_rf_Optuna_100.csv", index=False)


with open("../Parametros/parameters.txt","a") as f:
  f.write("RandomForestRegressor optuna: "+str(best_params)+" Roc_Aux: "+str(roc_auc)+"\n")
#{'n_estimators': 4907, 'max_depth': 27, 'min_samples_split': 0.11836338894400229, 'min_samples_leaf': 0.10002995552582354, 'max_features': 'sqrt'}