from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, make_scorer

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


# Define el espacio de búsqueda de hiperparámetros
param_dist = {
    'n_estimators': [100,200,300,400,500,1000,2000,3000],
    'max_depth': [20,30,50],
    'min_samples_split': [6,8,10],
    'min_samples_leaf': [5,7,9,10],
    'max_features': ['sqrt', 'log2']
}

# Crea una instancia del modelo
model = RandomForestRegressor()

# Realiza una búsqueda aleatoria utilizando la métrica personalizada
grid_search = GridSearchCV(model, param_grid=param_dist, cv=3, scoring='roc_auc')
grid_search.fit(X_train, y_train)
# Ajusta la búsqueda a tus datos
random_search=RandomizedSearchCV(model, param_distributions=param_dist,cv=3, scoring='roc_auc')
random_search.fit(X_train, y_train)

# Obtiene los mejores hiperparámetros
best_params = random_search.best_params_
s=(f"Mejores hiperparámetros random: {best_params}")
s=s.replace('\'','')
s=s.replace(':','=')
print(s)
best_model = random_search.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

roc_auc = roc_auc_score(y_test, y_pred)
print(f'ROC AUC Score: {roc_auc}')
df_test["Claim"] = best_model.predict(df_test[features_selected])
df_test[["Customer Id", "Claim"]].to_csv("../Submits/submit_test_rf_Rand.csv", index=False)

#best_params = grid_search.best_params_
s=(f"Mejores hiperparámetros grid: {best_params}")
s=s.replace('\'','')
s=s.replace(':','=')
print(s)
# Entrena el modelo final con los mejores hiperparámetros
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

roc_auc = roc_auc_score(y_test, y_pred)
print(f'ROC AUC Score: {roc_auc}')
df_test["Claim"] = best_model.predict(df_test[features_selected])
df_test[["Customer Id", "Claim"]].to_csv("../Submits/submit_test_rf_Grid.csv", index=False)
