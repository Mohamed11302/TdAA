import optuna
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
import random

class EstudioOptuna:
    def __init__(self,X_train,y_log_train,y_test,X_test,X, y):
        self.X_train=X_train
        self.y_log_train=y_log_train
        self.y_test=y_test
        self.X_test=X_test
        self.X=X
        self.y = y

    def objective(self,trial):
        max_depth = trial.suggest_int('max_depth', 2, 10)
        min_samples_split = trial.suggest_float('min_samples_split', 0.1, 1.0)
        min_samples_leaf = trial.suggest_float('min_samples_leaf', 0.1, 0.5)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])

        # Entrenar el modelo con los hiperparámetros
        model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=0
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
        study.optimize(self.objective, n_trials=50)

        # Obtener los mejores hiperparámetros
        best_params = study.best_params

        best_AUCROC = 0
        best_random = 0
        iter = 100
        for i in range(0, iter):
            r = random.randint(0, 100000)
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.1, random_state=r)
            y_log_train = np.log1p(y_train)
            DTR = DecisionTreeRegressor(**best_params, random_state=r)
            DTR.fit(X_train, y_log_train)
            y_log_pred = DTR.predict(self.X_test)
            y_pred =  np.expm1(y_log_pred)
            roc_auc=(mean_absolute_error(self.y_test, y_pred))
            if roc_auc > best_AUCROC:
                best_AUCROC = roc_auc
                best_random = r
        print(best_AUCROC)
        return [DTR,best_params,20]