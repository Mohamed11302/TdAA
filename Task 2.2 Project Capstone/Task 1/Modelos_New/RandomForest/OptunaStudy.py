import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
import random

class EstudioOptuna:
    def __init__(self,X_train,y_log_train,y_test,X_test,X, y,n):
        self.X_train=X_train
        self.y_log_train=y_log_train
        self.y_test=y_test
        self.X_test=X_test
        self.X=X
        self.y = y
        self.n = n

    def objective(self,trial):
        n_estimators = trial.suggest_int('n_estimators', self.n[0], self.n[1])

        # Entrenar el modelo con los hiperparámetros
        model = RandomForestRegressor(
        n_estimators=n_estimators
    )
        model.fit(self.X_train, self.y_log_train)

        # Predecir en el conjunto de validación
        y_log_pred = model.predict(self.X_test)
        y_pred =  np.expm1(y_log_pred)
        score=(mean_absolute_error(self.y_test, y_pred))
        print("MAE: "+str(score))
       

        return score

    def estudio(self):
        # Crear un estudio de Optuna y ejecutar la optimización
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=5)

        # Obtener los mejores hiperparámetros
        best_params = study.best_params

        best_AUCROC = 0
        best_random = 0
        iter = 5
        for i in range(0, iter):
            print(i)
            r = random.randint(0, 100000)
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.1, random_state=r)
            y_log_train = np.log1p(y_train)
            DTR = RandomForestRegressor(**best_params, random_state=r)
            DTR.fit(X_train, y_log_train)
            y_log_pred = DTR.predict(self.X_test)
            y_pred =  np.expm1(y_log_pred)
            roc_auc=(mean_absolute_error(self.y_test, y_pred))
            if roc_auc > best_AUCROC:
                best_AUCROC = roc_auc
                best_random = r
        print(best_AUCROC)
        return [DTR,best_params,20]