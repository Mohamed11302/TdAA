
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,StackingRegressor
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import pandas as pd
import numpy as np
import OptunaStudy
import OptunaStudy_Decision_tree
from sklearn.model_selection import RandomizedSearchCV
from random import randint


#loading the data
df = pd.read_csv("ObtencionDatos/train.csv")
df_test = pd.read_csv("ObtencionDatos/test.csv")


df.dropna(axis=0, how='any', subset=None, inplace=True)
df_OneHot = pd.get_dummies(df[['Gender','MaritalStatus', 'PartTimeFullTime']])
df_full = pd.concat([df, df_OneHot], axis = 1)
#transformed_target = np.log1p(original_target)

#features_selected = ['WeeklyWages', 'HoursWorkedPerWeek', 'InitialIncurredCalimsCost']
features_selected = ['Age',
 'DependentChildren',
 'DependentsOther',
 'WeeklyWages',
 'HoursWorkedPerWeek',
 'DaysWorkedPerWeek',
 'InitialIncurredCalimsCost',
 'Gender_F',
 'Gender_M',
 'MaritalStatus_M',
 'MaritalStatus_S',
 'MaritalStatus_U',
 'PartTimeFullTime_F',
 'PartTimeFullTime_P']

# more cleaning to get x and y labels
y = df['UltimateIncurredClaimCost']
X = df_full[features_selected]
# Divide tus datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
X_train.reset_index(drop = True, inplace = True)

#transform the UltimateClaimCost
y_log_train = np.log1p(y_train)
# Rebuild the model and made predictions
regr = RandomForestRegressor(n_estimators = 157)
regr.fit(X_train, y_log_train)
y_log_pred = regr.predict(X_test)
y_pred =  np.expm1(y_log_pred)
print ("MAE:", metrics.mean_absolute_error(y_test, y_pred))
print ("R^2:", metrics.r2_score(y_test, y_pred))
print("Ahora con Optuna...")

estimators=[]
for i in range(1,400):
    estimators.append(i)
param_dist = {
    'n_estimators': estimators,
    'max_depth': [20,30,50],
    'min_samples_split': [6,8,10],
    'min_samples_leaf': [5,7,9,10],
    'max_features': ['sqrt', 'log2']
}

# Construir el modelo de Random Forest
rf_model = RandomForestRegressor(random_state=42)

# Configurar RandomizedSearchCV
random_search = RandomizedSearchCV(
    rf_model, param_distributions=param_dist, n_iter=50,
    cv=5, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1
)

rf_model2 = RandomForestRegressor(random_state=42)

# Configurar RandomizedSearchCV
random_search2 = RandomizedSearchCV(
    rf_model2, param_distributions=param_dist, n_iter=50,
    cv=5, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1
)

base_models = [
    ('random_forest_1', random_search),
    ('random_forest_2', random_search2),
]

# Definir el meta-modelo
meta_model = RandomForestRegressor()

# Definir el modelo de conjunto
ensemble_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
X_train.reset_index(drop = True, inplace = True)
y_log_train = np.log1p(y_train)
ensemble_model.fit(X_train, y_log_train)
y_log_pred = ensemble_model.predict(X_test)
y_pred =  np.expm1(y_log_pred)
with open('Modelos_New/RandomForest/Parametros.txt','a') as f:
    f.write("Parametros: "+str("x")+" random_state: "+str(42)+" MAE: "+str(metrics.mean_absolute_error(y_test, y_pred))+"\n")
    f.close()
print ("MAE:", metrics.mean_absolute_error(y_test, y_pred))
print ("R^2:", metrics.r2_score(y_test, y_pred))

