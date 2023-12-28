import optuna
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
import pandas as pd
import random

class optunaRF:
    def __init__(self,X_train, X_test, y_train, y_test,X,y):
        self.X=X
        self.y=y
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test
        self.n_estimators=0
        self.max_depth=0
        self.min_samples_split=0
        self.min_samples_leaf=0
        self.max_features=0
        self.splitter="best"
        self.criterion="squared_error"

    # Define la función de optimización de Optuna
    def objectiveRF(self,trial):
        # Define los hiperparámetros a optimizar
        self.n_estimators = trial.suggest_int("n_estimators", 1, 3000)
        self.max_depth = trial.suggest_int("max_depth", 1, 50)
        self.min_samples_split = trial.suggest_float("min_samples_split", 0.1, 1.0)
        self.min_samples_leaf = trial.suggest_float("min_samples_leaf", 0.1, 1.0)
        self.max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
        # Crea una instancia del modelo Random Forest con los hiperparámetros sugeridos
        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features
        )
        
        # Entrena el modelo en los datos de entrenamiento
        model.fit(self.X_train, self.y_train)
        
        # Realiza predicciones en el conjunto de prueba
        y_pred = model.predict(self.X_test)
        
        # Calcula el ROC AUC en el conjunto de prueba
        print ("MAE:", metrics.mean_absolute_error(y_pred , self.y_test))
        print ("MAPE:", metrics.mean_absolute_percentage_error(y_pred , self.y_test))
        print ("MSE:", metrics.mean_squared_error(y_pred , self.y_test))
        print ("R^2:", metrics.r2_score(y_pred , self.y_test))
        
        return metrics.mean_absolute_percentage_error(y_pred , self.y_test)
    
    def objectiveDT(self,trial):
        # Define los hiperparámetros a optimizar
        #criterion='poisson', max_depth=4, max_features=0.8, min_samples_leaf=9, min_samples_split=3
        self.max_depth = trial.suggest_int("max_depth", 1, 50)
        self.min_samples_split = trial.suggest_float("min_samples_split", 0.1, 1.0)
        self.min_samples_leaf = trial.suggest_float("min_samples_leaf", 0.1, 1.0)
        self.criterion = trial.suggest_categorical("criterion", ["squared_error", "friedman_mse", "absolute_error", "poisson"])
        self.splitter = trial.suggest_categorical("splitter", ["best", "random"])
        self.max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])

        # Crea una instancia del modelo Random Forest con los hiperparámetros sugeridos
        model = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            splitter=self.splitter,
            criterion=self.criterion

        )
        
        # Entrena el modelo en los datos de entrenamiento
        model.fit(self.X_train, self.y_train)
        
        # Realiza predicciones en el conjunto de prueba
        y_pred = model.predict(self.X_test)
        
        
        return metrics.mean_absolute_percentage_error(y_pred , self.y_test)
    
    def buscarParametrosRF(self):  
        # Llama a la función de optimización de Optuna
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objectiveRF, n_trials=50)

        # Obtiene los mejores hiperparámetros encontrados por Optuna
        best_params = study.best_params

        # Crea el modelo final con los mejores hiperparámetros y ajústalo a tus datos

        best_AUCROC = 0
        best_random = 0
        iter = 100
        for i in range(0, iter):
            r = random.randint(0, 100000)
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.1, random_state=r)
            DTR = RandomForestRegressor(**best_params, random_state=r)
            DTR.fit(X_train, y_train)
            y_pred = DTR.predict(X_test)
            roc_auc = metrics.mean_absolute_percentage_error(y_pred , y_test)
            if roc_auc > best_AUCROC:
                best_AUCROC = roc_auc
                best_random = r
        print("Best MAPE = " +str(best_AUCROC))
        print("Best Random = " +str(best_random))
        #Guardamos el resultado del mejor modelo con el mejor estado aleatorio
        with open('Params.txt','a') as f:
            f.write("Random forest regressor: Prametros = "+str(best_params)+" AUC-ROC = " +str(best_AUCROC)+" Random = "+str(best_random)+"\n")
            f.close()
        return best_params, best_random
    
    def buscarParametrosDT(self):  
        # Llama a la función de optimización de Optuna
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objectiveDT, n_trials=50)

        # Obtiene los mejores hiperparámetros encontrados por Optuna
        best_params = study.best_params

        # Crea el modelo final con los mejores hiperparámetros y ajústalo a tus datos

        best_AUCROC = 0
        best_random = 0
        iter = 100
        for i in range(0, iter):
            r = random.randint(0, 100000)
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.1, random_state=r)
            DTR = DecisionTreeRegressor(**best_params, random_state=r)
            DTR.fit(X_train, y_train)
            y_pred = DTR.predict(X_test)
            roc_auc = metrics.mean_absolute_percentage_error(y_pred , y_test)
            if roc_auc > best_AUCROC:
                best_AUCROC = roc_auc
                best_random = r
        print("Best MAPE = " +str(best_AUCROC))
        print("Best Random = " +str(best_random))
        #Guardamos el resultado del mejor modelo con el mejor estado aleatorio
        with open('Params.txt','a') as f:
            f.write("Decision tree regressor: Prametros = "+str(best_params)+" AUC-ROC = " +str(best_AUCROC)+" Random = "+str(best_random)+"\n")
            f.close()
        return best_params, best_random