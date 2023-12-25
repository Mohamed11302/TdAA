import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D, Input, Concatenate, Flatten
from tensorflow.keras.models import Sequential
from keras.models import Model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from keras.optimizers import Adam
from sklearn.feature_selection import f_regression
from scipy.sparse import csr_matrix
import autokeras as ak

def crearModelo_RandomForest(df):
    X_text = df[['Claim_Body_Parts', 'Claim_Injuries']]
    y = df['UltimateIncurredClaimCost']

    # Dividir datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.33, random_state=42)

    # TF-IDF Vectorizer para convertir texto a características numéricas
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train['Claim_Body_Parts'] + ' ' + X_train['Claim_Injuries'])
    X_test_tfidf = tfidf_vectorizer.transform(X_test['Claim_Body_Parts'] + ' ' + X_test['Claim_Injuries'])

    # Selección de características utilizando SelectKBest
    k_best_selector = SelectKBest(score_func=f_regression, k=100)
    X_train_selected = k_best_selector.fit_transform(X_train_tfidf, y_train)
    X_test_selected = k_best_selector.transform(X_test_tfidf)

    # Entrenar un modelo de Random Forest
    random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
    random_forest_model.fit(X_train_selected, y_train)

    # Hacer predicciones en el conjunto de prueba
    y_pred = random_forest_model.predict(X_test_selected)
    print ("MAE:", metrics.mean_absolute_error(y_pred , y_test))
    print ("MAPE:", metrics.mean_absolute_percentage_error(y_pred , y_test))
    print ("MSE:", metrics.mean_squared_error(y_pred , y_test))
    print ("R^2:", metrics.r2_score(y_pred , y_test))




def crearmodelo_NN(df):
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)
    selected_features = ["Claim_Body_Parts", "Claim_Injuries"]
    X_text = df[selected_features]
    y = df["UltimateIncurredClaimCost"]
    max_features = 10000
    embed_dim = 100
    maxlen = 10

    # Convertimos el texto en secuencias de enteros
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(X_text["Claim_Body_Parts"] + X_text["Claim_Injuries"])

    sequences_body_parts = tokenizer.texts_to_sequences(X_text["Claim_Body_Parts"])
    sequences_injuries = tokenizer.texts_to_sequences(X_text["Claim_Injuries"])

    # Aseguramos que todas las secuencias tengan la misma longitud
    data_body_parts = pad_sequences(sequences_body_parts, maxlen=maxlen)
    data_injuries = pad_sequences(sequences_injuries, maxlen=maxlen)
    print("-")
    print(data_body_parts)
    print(data_injuries)
    print("--")
    # Creamos la capa de incrustación de palabras
    embedding_layer = Embedding(max_features, embed_dim, input_length=maxlen)

    # Creamos la arquitectura de la red neuronal
    input_body_parts = Input(shape=(maxlen,), dtype='int32')
    embedded_body_parts = embedding_layer(input_body_parts)
    input_injuries = Input(shape=(maxlen,), dtype='int32')
    embedded_injuries = embedding_layer(input_injuries)

    # Concatenamos las dos entradas
    concatenated = Concatenate()([Flatten()(embedded_body_parts), Flatten()(embedded_injuries)])
    # Añadimos capas densas adicionales
    dense_layer1 = Dense(64, activation='relu')(concatenated)
    dense_layer2 = Dense(32, activation='relu')(dense_layer1)
    dense_layer3 = Dense(16, activation='relu')(dense_layer2)
    dense_layer4 = Dense(8, activation='relu')(dense_layer3)
    # Capa de salida para la regresión
    output = Dense(1, activation='linear')(dense_layer4)

    # Compilamos el modelo
    model = Model(inputs=[input_body_parts, input_injuries], outputs=output)
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

    # Dividimos los datos en conjuntos de entrenamiento y prueba
    X_train_body_parts, X_test_body_parts, X_train_injuries, X_test_injuries, y_train, y_test = train_test_split(
        data_body_parts, data_injuries, y, test_size=0.33, random_state=42
    )

    # Entrenamos el modelo
    model.fit([X_train_body_parts, X_train_injuries], y_train, epochs=10, batch_size=32, validation_split=0.2, shuffle=False)

    # Evaluamos el modelo en el conjunto de prueba
    y_pred = model.predict([X_test_body_parts, X_test_injuries])
    print("MAE:", metrics.mean_absolute_error(y_pred, y_test))
    print("MSE:", metrics.mean_squared_error(y_pred, y_test))
    print("R^2:", metrics.r2_score(y_test, y_pred))

def crearmodelo_NN_tfidf(df):
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)
    selected_features = ["Claim_Body_Parts", "Claim_Injuries"]
    X_text = df[selected_features]
    y = df["UltimateIncurredClaimCost"]
    max_features=1000
    maxlen = 10
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(X_text["Claim_Body_Parts"] + X_text["Claim_Injuries"])

    sequences_body_parts = tokenizer.texts_to_sequences(X_text["Claim_Body_Parts"])
    sequences_injuries = tokenizer.texts_to_sequences(X_text["Claim_Injuries"])

    data_body_parts = pad_sequences(sequences_body_parts, maxlen=maxlen)
    data_injuries = pad_sequences(sequences_injuries, maxlen=maxlen)

    X = np.concatenate([data_body_parts, data_injuries], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    reg = ak.StructuredDataRegressor(max_trials=10, overwrite=True)

    reg.fit(X_train, y_train, epochs=10)

    # Evaluamos el modelo en el conjunto de prueba
    y_pred = reg.predict(X_test)
    print("MAE:", metrics.mean_absolute_error(y_test, y_pred))
    print("MSE:", metrics.mean_squared_error(y_test, y_pred))
    print("R^2:", metrics.r2_score(y_test, y_pred))



def crearmodelo_NN_BODYPARTS(df):
    seed = 42
    tf.random.set_seed(seed)
    selected_features = ["Claim_Body_Parts"]
    X_text = df[selected_features]
    y = df["UltimateIncurredClaimCost"]
    max_features = 1000
    embed_dim = 100
    maxlen = 10

    # Convertimos el texto en secuencias de enteros
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(X_text["Claim_Body_Parts"])

    sequences_body_parts = tokenizer.texts_to_sequences(X_text["Claim_Body_Parts"])

    # Aseguramos que todas las secuencias tengan la misma longitud
    data_body_parts = pad_sequences(sequences_body_parts, maxlen=maxlen)
    embedding_layer = Embedding(max_features, embed_dim, input_length=maxlen)

    # Creamos la arquitectura de la red neuronal
    input_body_parts = Input(shape=(maxlen,), dtype='int32')
    embedded_body_parts = embedding_layer(input_body_parts)

    # Concatenamos las dos entradas
    concatenated = Concatenate()([Flatten()(embedded_body_parts)])
    # Añadimos capas densas adicionales
    dense_layer1 = Dense(64, activation='relu')(concatenated)
    dense_layer2 = Dense(32, activation='relu')(dense_layer1)
    dense_layer3 = Dense(16, activation='relu')(dense_layer2)
    # Capa de salida para la regresión
    output = Dense(1, activation='linear')(dense_layer3)

    # Compilamos el modelo
    model = Model(inputs=[input_body_parts], outputs=output)
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

    # Dividimos los datos en conjuntos de entrenamiento y prueba
    X_train_body_parts, X_test_body_parts, y_train, y_test = train_test_split(
        data_body_parts, y, test_size=0.33, random_state=42
    )

    # Entrenamos el modelo
    model.fit([X_train_body_parts], y_train, epochs=10, batch_size=32, validation_split=0.2, shuffle=True)

    # Evaluamos el modelo en el conjunto de prueba
    y_pred = model.predict([X_test_body_parts])
    print("MAE:", metrics.mean_absolute_error(y_pred, y_test))
    print("MSE:", metrics.mean_squared_error(y_pred, y_test))
    print("R^2:", metrics.r2_score(y_test, y_pred))



def crearmodelo_NN_base(df):
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)
    selected_features = ["ClaimDescription"]
    X_text = df[selected_features]
    y = df["UltimateIncurredClaimCost"]
    max_features = 1000
    embed_dim = 100
    maxlen = 10

    # Convertimos el texto en secuencias de enteros
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(X_text["ClaimDescription"])

    sequences_body_parts = tokenizer.texts_to_sequences(X_text["ClaimDescription"])

    # Aseguramos que todas las secuencias tengan la misma longitud
    data_body_parts = pad_sequences(sequences_body_parts, maxlen=maxlen)
    # Creamos la capa de incrustación de palabras
    embedding_layer = Embedding(max_features, embed_dim, input_length=maxlen)

    # Creamos la arquitectura de la red neuronal
    input_body_parts = Input(shape=(maxlen,), dtype='int32')
    embedded_body_parts = embedding_layer(input_body_parts)

    # Concatenamos las dos entradas
    concatenated = Concatenate()([Flatten()(embedded_body_parts)])
    # Añadimos capas densas adicionales
    dense_layer1 = Dense(64, activation='relu')(concatenated)
    dense_layer2 = Dense(32, activation='relu')(dense_layer1)
    dense_layer3 = Dense(16, activation='relu')(dense_layer2)
    # Capa de salida para la regresión
    output = Dense(1, activation='linear')(dense_layer3)

    # Compilamos el modelo
    model = Model(inputs=[input_body_parts], outputs=output)
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

    # Dividimos los datos en conjuntos de entrenamiento y prueba
    X_train_body_parts, X_test_body_parts, y_train, y_test = train_test_split(
        data_body_parts, y, test_size=0.33, random_state=42
    )

    # Entrenamos el modelo
    model.fit([X_train_body_parts], y_train, epochs=10, batch_size=32, validation_split=0.2, shuffle=False)

    # Evaluamos el modelo en el conjunto de prueba
    y_pred = model.predict([X_test_body_parts])
    print("MAE:", metrics.mean_absolute_error(y_pred, y_test))
    print("MSE:", metrics.mean_squared_error(y_pred, y_test))
    print("R^2:", metrics.r2_score(y_test, y_pred))



def crearmodelo_NN_base_CNN(df):
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)
    selected_features = ["ClaimDescription"]
    X_text = df[selected_features]
    y = df["UltimateIncurredClaimCost"]
    max_features = 1000
    embed_dim = 100
    maxlen = 10
    