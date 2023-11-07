import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
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
#splitting data into 30% test and 70% train
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 42)

#predicting the claim
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

df_test["Claim"] = clf.predict(df_test[features_selected])

df_test["Claim"].value_counts()

#df_test[["Customer Id", "Claim"]].to_csv("sub.csv", index = False)

features_selected = ['Insured_Period', 'Residential','Building_Fenced', 'Settlement', 'Building Dimension', 'Building_Type']
X = df[features_selected]
y = df.Claim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
priors = np.array([0.43, 0.56]) #Mejores resultados hasta ahora
clf = GaussianNB(var_smoothing=1e-7, priors=priors)
clf.fit(X_train, y_train)
df_test["Claim"] = clf.predict(df_test[features_selected])
y_pred = clf.predict(X_test)
roc_auc = roc_auc_score(y_test, y_pred)
print(f'ROC AUC Score: {roc_auc}')
df_test[["Customer Id", "Claim"]].to_csv("../Submits/submit_test_01.csv", index = False)

