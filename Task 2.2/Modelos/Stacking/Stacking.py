from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,StackingRegressor
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import pandas as pd
import OptunaRF


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

#Optimizamos los modelos fuera de este archivo
optRF=OptunaRF.optunaRF(X_train, X_test, y_train, y_test,X,y)
best_paramsRF,best_randomRF=optRF.buscarParametrosRF()
best_paramsDT,best_randomDT=optRF.buscarParametrosDT()
# Definir los modelos base
base_models = [
    ('random_forest', RandomForestRegressor(**best_paramsRF,random_state=best_randomRF)),
    ('decision_tree', DecisionTreeRegressor(**best_paramsDT,random_state=best_randomDT)),
    ('linear_regression', LinearRegression()),
    ('gradient_boosting', GradientBoostingRegressor()),
]

# Definir el meta-modelo
meta_model = LinearRegression()

# Definir el modelo de conjunto
ensemble_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

# Ajustar el modelo a los datos
ensemble_model.fit(X_train, y_train)
y_pred = ensemble_model.predict(X_test)
roc_auc = metrics.r2_score(y_pred , y_test)
print("ROC_AUC: "+str(roc_auc))
print ("MAE:", metrics.mean_absolute_error(y_pred , y_test))
print ("MAPE:", metrics.mean_absolute_percentage_error(y_pred , y_test))
print ("MSE:", metrics.mean_squared_error(y_pred , y_test))
print ("R^2:", metrics.r2_score(y_pred , y_test))
df_test["Claim"] = ensemble_model.predict(df_test[features_selected])
df_test[["Customer Id", "Claim"]].to_csv("Submits/submit_test_Stacking.csv", index = False)