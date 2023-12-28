from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,StackingRegressor
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from scikeras.wrappers import KerasRegressor



def build_neural_network(input):
    model = Sequential()
    model.add(Dense(256, input_dim=input.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, input_dim=input.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, input_dim=input.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    optimizer = Adam(learning_rate=0.001)
    h = tf.keras.losses.LogCosh()
    model.compile(loss=h, optimizer=optimizer, metrics=['mae'])
    print("Nuevo")
    return model

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

# Define los modelos base
# Crear el modelo

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
dt_model = DecisionTreeRegressor(max_depth=4,random_state=42)
rn_model = KerasRegressor(model=lambda: build_neural_network(X_train), epochs=5, batch_size=32, verbose=0)
# ...
# Define el modelo de ensamble (usando regresión lineal como meta-regresor)
ensemble_model = StackingRegressor(
    estimators=[('rf', rf_model), ('gb', gb_model),('dt', dt_model),('rn',rn_model)],
    final_estimator=dt_model
)

# Entrenamiento del modelo de ensamble
ensemble_model.fit(X_train, y_log_train)

# Predicciones
y_log_pred = ensemble_model.predict(X_test)
y_pred =  np.expm1(y_log_pred)
# Evaluación del rendimiento
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'R^2 Score: {r2}')
