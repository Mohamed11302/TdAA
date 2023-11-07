from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn import metrics
import pandas as pd

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Define el espacio de búsqueda de hiperparámetros para la búsqueda aleatoria
param_dist_random = {
    'n_estimators': range(5000),
    'max_depth': range(50),
    'min_samples_split': range(2,10),
    'min_samples_leaf': range(1,10),
    'max_features': ['sqrt', 'log2']
}

# Crea una instancia del modelo
model = RandomForestRegressor(random_state=42)

# Realiza una búsqueda aleatoria utilizando la métrica personalizada
random_search = RandomizedSearchCV(model, param_distributions=param_dist_random, n_iter=10, cv=3, scoring='roc_auc', random_state=42)

# Ajusta la búsqueda aleatoria a tus datos
random_search.fit(X_train, y_train)

# Obtiene los mejores hiperparámetros de la búsqueda aleatoria
best_params_random = random_search.best_params_

# Define el espacio de búsqueda de hiperparámetros para la búsqueda en cuadrícula
param_grid_grid = {
    'n_estimators': [best_params_random['n_estimators'], best_params_random['n_estimators'] + 50, best_params_random['n_estimators'] + 100],
    'max_depth': [best_params_random['max_depth']],
    'min_samples_split': [best_params_random['min_samples_split']],
    'min_samples_leaf': [best_params_random['min_samples_leaf']],
    'max_features': [best_params_random['max_features']]
}

# Realiza una búsqueda en cuadrícula utilizando la métrica personalizada
grid_search = GridSearchCV(model, param_grid=param_grid_grid, cv=3, scoring='roc_auc')

# Ajusta la búsqueda en cuadrícula a tus datos
grid_search.fit(X_train, y_train)
for i, score in enumerate(grid_search.cv_results_['mean_test_score']):
    print(f"Iteración {i+1} de Grid Search: ROC AUC = {score}")
# Obtiene los mejores hiperparámetros de la búsqueda en cuadrícula
best_params_grid = grid_search.best_params_

# Combina los mejores hiperparámetros encontrados
best_params_combined = {
    'n_estimators': best_params_grid['n_estimators'],
    'max_depth': best_params_random['max_depth'],
    'min_samples_split': best_params_random['min_samples_split'],
    'min_samples_leaf': best_params_random['min_samples_leaf'],
    'max_features': best_params_random['max_features']
}

# Entrena el modelo final con los mejores hiperparámetros combinados
best_model = RandomForestRegressor(**best_params_combined, random_state=42)
best_model.fit(X_train, y_train)

# Calcula el ROC AUC en el conjunto de prueba
y_pred = best_model.predict(X_test)

roc_auc = roc_auc_score(y_test, y_pred)
print(f'ROC AUC Score: {roc_auc}')

df_test["Claim"] = best_model.predict(df_test[features_selected])
#print(best_model.get_params())
best_params = best_params_combined
s=(f"Mejores hiperparámetros: {best_params}")
s=s.replace('\'','')
s=s.replace(':','=')
print(s)

df_test[["Customer Id", "Claim"]].to_csv("../Submits/submit_test_rf_Comb.csv", index=False)