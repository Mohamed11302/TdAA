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
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler

import datetime


#loading the data
df = pd.read_csv("../Task 3/clustering_result2.csv")

print(df)

df.dropna(axis=0, how='any', subset=None, inplace=True)
times=[]
years=[]
months=[]
daysPassed=[]
cont=0
for i in df['DateTimeOfAccident']:
    times.append(datetime.datetime(int(i[0:4]), int(i[5:7]), int(i[8:10])))
    
for i in df['DateReported']:
    t=times.pop(0)
    daysPassed.append((datetime.datetime(int(i[0:4]), int(i[5:7]), int(i[8:10]))-t).days)
    years.append(datetime.datetime(int(i[0:4]), int(i[5:7]), int(i[8:10])).year)
    months.append(datetime.datetime(int(i[0:4]), int(i[5:7]), int(i[8:10])).month%4)


yearsN= years.copy()
for i in range(len(yearsN)):
    yearsN[i] -= min(years)


df['TimePassed']=daysPassed
df['years']=yearsN
df['months']=months
df_OneHot = pd.get_dummies(df[['Gender','MaritalStatus', 'PartTimeFullTime']])
df_full = pd.concat([df, df_OneHot], axis = 1)


# Calcula la matriz de correlación
correlation_matrix = df_full[['TimePassed','years','UltimateIncurredClaimCost','months']].corr()

# Visualiza la matriz de correlación
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación')
plt.show()


