from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,StackingRegressor
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler


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
y_log_train = np.log1p(y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear el modelo
model = Sequential()
'''model.add(Dense(1024, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dropout(0.5))'''
model.add(Dense(256, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))

optimizer = Adam(learning_rate=0.001)
h = tf.keras.losses.LogCosh()
# Compilar el modelo
model.compile(loss=h, optimizer=optimizer, metrics=['mae'])

# Entrenar el modelo
model.fit(X_train_scaled, y_log_train, epochs=200, batch_size=32, validation_split=0.2)

# Evaluar el modelo en el conjunto de prueba
y_log_pred = model.predict(X_test_scaled)

y_pred =  np.expm1(y_log_pred)
# Calcular métricas de rendimiento
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'R^2 Score: {r2}')