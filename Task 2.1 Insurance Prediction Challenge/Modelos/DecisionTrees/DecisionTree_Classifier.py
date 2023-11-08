import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import random



def report(results, n_top):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def evaluate_model(y_test, y_pred, y_pred_proba, classes):
    # Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
    # Print the classification report
    print(classification_report(y_test, y_pred))
    # roc
    y_score = y_pred_proba[:, 1]
    ROC_AUC = roc_auc_score(y_test, y_score)
    print('ROC AUC : {:.4f}'.format(ROC_AUC))

def limpieza(df, df_test):
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

    df = df.drop("Geo_Code", axis = 1)
    df_test = df_test.drop("Geo_Code", axis = 1)

    #change to numerical values
    df.Building_Painted.replace(('N','V'),(1,0), inplace = True)
    df_test.Building_Painted.replace(('N','V'),(1,0), inplace = True)
    df.Building_Fenced.replace(('N','V'),(1,0), inplace = True)
    df_test.Building_Fenced.replace(('N','V'),(1,0), inplace = True)
    df.Garden.replace(('V','O'),(1,0), inplace = True)
    df_test.Garden.replace(('V','O'),(1,0), inplace = True)
    df.Settlement.replace(('U','R'),(1,0), inplace = True)
    df_test.Settlement.replace(('U','R'),(1,0), inplace = True)
    return df, df_test

def Select_Best_Features(df, num_features):
    features_selected = df.select_dtypes(include=['int64', 'float64']).columns.tolist()[:-1] 
    selector = SelectKBest(score_func=f_classif, k=num_features)
    X = df[features_selected]
    y = df.Claim
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    selected_features = np.array(features_selected)[selector.get_support()]
    #print("Selected features:", selected_features)

    return selected_features



def Grid_Modelo(df):
    #features_selected = ['Insured_Period', 'Residential','Building_Fenced', 'Garden','Settlement', 'Building Dimension', 'Building_Type']
    features_selected = df.select_dtypes(include=['int64', 'float64']).columns.tolist()[:-1]    
    X = df[features_selected]
    y = df.Claim
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = DecisionTreeClassifier()
    param_dist = {
    'n_neighbors': (1, 3, 5, 7, 10, 14, 18, 21),    
    'p': [1, 2], 
    'weights': ['uniform', 'distance'],
    #'metric': ['euclidean', 'chebyshev'],
    #'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': (15, 17, 20, 23, 25),
    }
    param_dist = {
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "max_depth": ["Integer", "None"],
        "min_samples_split": (0.1, 0.3, 0.5, 0.7, 0.9),
        "min_samples_leaf": (0.1, 0.3, 0.5, 0.7, 0.9),
        "max_features": ["Integer", "Float", "auto", "sqrt", "log2", "None"],
        "max_leaf_nodes": ["Integer", "None"],
        "class_weight": ["balanced", "None", "Dictionary"],
        "random_state": 42,  #Its an integer number
        #"min_impurity_decrease": "Float",
        "presort": ["True", "False"],
        #"ccp_alpha": "Float"
    }
    grid_clf = GridSearchCV(estimator = clf, param_grid= param_dist, cv = 10, scoring='roc_auc', verbose=1, n_jobs=-1)
    grid_clf.fit(X = X_train, y = y_train)
    report(grid_clf.cv_results_, n_top = 5)
    print(grid_clf.best_estimator_)
    best_grid = grid_clf.best_estimator_
    # fit and predict
    best_grid.fit( X = X_train, y = y_train)
    y_pred = best_grid.predict(X_test)
    y_pred_proba = best_grid.predict_proba(X_test)
    evaluate_model(y_test, y_pred, y_pred_proba, best_grid.classes_)

def MejorModelo(df, df_test):
    ####################################################################
    #               MEJOR MODELO POR AHORA
    # 0.643453991
    # DecisionTreeClassifier(class_weight='balanced', criterion='entropy', max_depth=5, max_features=0.6, min_samples_leaf=12, min_samples_split=3)
    # features_selected = df.select_dtypes(include=['int64', 'float64']).columns.tolist()[:-1]   
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=5747)
    ####################################################################
    #features_selected = ['Insured_Period', 'Residential','Settlement', 'Building Dimension', 'Building_Type']
    features_selected = df.select_dtypes(include=['int64', 'float64']).columns.tolist()[:-1]    
    X = df[features_selected]
    y = df.Claim
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=5747)
    DTC = DecisionTreeClassifier(max_depth=5, max_features=0.7000000000000001, min_samples_leaf=14, min_samples_split=10)
    DTC.fit(X_train, y_train)
    df_test["Claim"] = DTC.predict(df_test[features_selected])
    df_test[["Customer Id", "Claim"]].to_csv("Submits/submit_test_DTC.csv", index = False)



def Randomized_Model(df):
    #features_selected = ['Insured_Period', 'Residential','Settlement', 'Building Dimension', 'Building_Type']
    features_selected = df.select_dtypes(include=['int64', 'float64']).columns.tolist()[:-1]    
    X = df[features_selected]
    y = df.Claim
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5747)

    clf = DecisionTreeClassifier()
    
    param_dist = {
    "criterion": ["gini", "entropy"],
    "splitter": ["best", "random"],
    "max_depth": [None] + list(np.arange(1, 20)),
    "min_samples_split": np.arange(2, 20),
    "min_samples_leaf": np.arange(1, 20),
    "class_weight": ["balanced", None],
    "max_features": ["sqrt", "log2", None] + list(np.arange(0.1, 1.1, 0.1)),
    }
    
    random_clf = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=5000, cv=30, scoring='roc_auc', n_jobs=-1)
    random_clf.fit(X_train, y_train)
    report(random_clf.cv_results_, n_top = 5)
    print(random_clf.best_estimator_)
    best_grid = random_clf.best_estimator_
    # fit and predict
    best_grid.fit( X = X_train, y = y_train)


def probar_random_state(df, max_random):
    features_selected = ['Insured_Period', 'Residential', 'Settlement', 'Building Dimension', 'Building_Type']
    X = df[features_selected]
    y = df.Claim
    r = random.randint(0, max_random)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=r)
    #print("Random: " + str(r))
    clf = DecisionTreeClassifier(class_weight='balanced', criterion='entropy', max_depth=5, max_features=0.6, min_samples_leaf=12, min_samples_split=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    roc_auc = metrics.roc_auc_score(y_test, y_pred)
    return r, roc_auc

def iteraciones_probar_rs(df, iteraciones, max_random):
    mejor_AUCROC = 0
    mejor_random = 0
    for i in range(0, iteraciones):
        r, roc_auc = probar_random_state(df, max_random)
        #print("Random = " + str(r))
        #print("AUC-ROC = " +str(roc_auc))
        if roc_auc > mejor_AUCROC:
            mejor_AUCROC = roc_auc
            mejor_random = r
    print("----")
    print("AUC-ROC = " +str(mejor_AUCROC))
    print("Random = " +str(mejor_random))


def main():
    #loading the data
    df=  pd.read_csv("data-science-nigeria-2019-challenge-1-insurance-prediction\\train_data.csv")
    df_test = pd.read_csv("data-science-nigeria-2019-challenge-1-insurance-prediction\\test_data.csv")
    df, df_test = limpieza(df, df_test)
    MejorModelo(df, df_test)



if __name__ == '__main__':
    main()