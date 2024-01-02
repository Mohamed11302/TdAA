
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
regr = RandomForestRegressor(n_estimators = 100)
regr.fit(X_train, y_log_train)
y_log_pred = regr.predict(X_test)
y_pred =  np.expm1(y_log_pred)
print ("MAE:", metrics.mean_absolute_error(y_test, y_pred))
print ("R^2:", metrics.r2_score(y_test, y_pred))
print("Ahora con Optuna...")

es=OptunaStudy.EstudioOptuna(X_train,y_log_train,y_test,X_test,X,y,[1,50])
[finalModel1,bestParams,bestRandom]=es.estudio()
es=OptunaStudy.EstudioOptuna(X_train,y_log_train,y_test,X_test,X,y,[50,100])
[finalModel2,bestParams,bestRandom]=es.estudio()
es=OptunaStudy.EstudioOptuna(X_train,y_log_train,y_test,X_test,X,y,[100,150])
[finalModel3,bestParams,bestRandom]=es.estudio()
es=OptunaStudy.EstudioOptuna(X_train,y_log_train,y_test,X_test,X,y,[150,200])
[finalModel4,bestParams,bestRandom]=es.estudio()

#es=OptunaStudy_Decision_tree.EstudioOptuna(X_train,y_log_train,y_test,X_test,X,y)
#[finalModelDT,bestParams,bestRandom]=es.estudio()


base_models = [
    ('random_forest_1', finalModel1),
    ('random_forest_2', finalModel2),
    ('random_forest_3', finalModel3),
    ('random_forest_4', finalModel4),
]

# Definir el meta-modelo
meta_model = RandomForestRegressor()


# Definir el modelo de conjunto
ensemble_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)


X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X, y, test_size=0.1, random_state=bestRandom)
y_log_train = np.log1p(y_train_new)
DTR = RandomForestRegressor(n_estimators = 184, random_state=bestRandom)
DTR.fit(X_train_new, y_log_train)
y_log_pred = DTR.predict(X_test)
y_pred =  np.expm1(y_log_pred)
roc_auc=(metrics.mean_absolute_error(y_test, y_pred))

with open('Modelos_New/RandomForest/Parametros.txt','a') as f:
    f.write("Parametros: "+str(bestParams)+" random_state: "+str(bestRandom)+" MAE: "+str(metrics.mean_absolute_error(y_test, y_pred))+"\n")
    f.close()
print ("MAE:", metrics.mean_absolute_error(y_test, y_pred))
print ("R^2:", metrics.r2_score(y_test, y_pred))

