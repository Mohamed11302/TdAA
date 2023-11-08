from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,StackingRegressor
from sklearn.metrics import roc_auc_score
import pandas as pd
import OptunaRF

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

features_selected = df.select_dtypes(include=['int64', 'float64']).columns.tolist()[:-1] 
#features_selected = ['Insured_Period', 'Residential', 'Settlement', 'Building Dimension', 'Building_Type']
X = df[features_selected]
y = df.Claim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

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
roc_auc = roc_auc_score(y_test, y_pred)
print("ROC_AUC: "+str(roc_auc))
df_test["Claim"] = ensemble_model.predict(df_test[features_selected])
df_test[["Customer Id", "Claim"]].to_csv("Submits/submit_test_Stacking.csv", index = False)