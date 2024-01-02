
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,StackingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
import pandas as pd
import numpy as np
import OptunaStudy
import OptunaStudy_Decision_tree
from sklearn.model_selection import RandomizedSearchCV
import random


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
regr = RandomForestRegressor(n_estimators = 184)
regr.fit(X_train, y_log_train)
y_log_pred = regr.predict(X_test)
y_pred =  np.expm1(y_log_pred)

best_AUCROC = 0
best_random = 0
iter = 5
for i in range(0, iter):
    print(i)
    r = random.randint(0, 100000)
    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X, y, test_size=0.1, random_state=r)
    y_log_train = np.log1p(y_train_new)
    DTR = RandomForestRegressor(n_estimators = 184, random_state=r)
    DTR.fit(X_train_new, y_log_train)
    y_log_pred = DTR.predict(X_test)
    y_pred =  np.expm1(y_log_pred)
    roc_auc=(mean_absolute_error(y_test, y_pred))
    if roc_auc > best_AUCROC:
        best_AUCROC = roc_auc
        best_random = r
print(best_AUCROC)


print ("MAE:", metrics.mean_absolute_error(y_test, y_pred))
print ("R^2:", metrics.r2_score(y_test, y_pred))