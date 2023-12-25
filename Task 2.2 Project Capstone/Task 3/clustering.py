import nltk 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import unidecode
nltk.download('wordnet')
nltk.download('stopwords')
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import plotly.express as px
porter=nltk.PorterStemmer()
import seaborn as sns
import sklearn
stopwords = set(stopwords.words("english"))
import numpy as np
from scipy import cluster
from sklearn import metrics
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, pairwise_distances
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
min_max_scaler = preprocessing.MinMaxScaler()
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model


def cargar_train():
    df = pd.read_csv("datasets/train.csv")
    columnas_necesarias = ['ClaimDescription', 'UltimateIncurredClaimCost']
    todas_las_columnas = df.columns.tolist()
    columnas_a_eliminar = [col for col in todas_las_columnas if col not in columnas_necesarias]
    df = df.drop(columns=columnas_a_eliminar)
    return df

def representar_datos(df): ### Utilizarlo cuando procese los datos para que se vea el resultado
    fig = px.scatter(df, x='ClaimDescription', y='UltimateIncurredClaimCost')#, color="label")
    fig.show()

def convert_tolower(df):
    for indice, fila in df.iterrows():
        cadena_original = fila["ClaimDescription"]
        df.at[indice, "ClaimDescription"] = cadena_original.lower()
    return df

def clean_text(df):
    df_nuevo = df.copy()

    # Itera sobre cada fila del dataframe
    for indice, fila in df_nuevo.iterrows():
        cadena_original = fila["ClaimDescription"]
        palabras = cadena_original.split()
        palabras = [porter.stem(palabra) for palabra in palabras]
        palabras_sin_repetir = list(dict.fromkeys(palabras))
        nueva_cadena = ' '.join(palabras_sin_repetir)
        nueva_cadena = word_tokenize(unidecode.unidecode(nueva_cadena))
        cadena_2 =  [w for w in nueva_cadena if not w in stopwords]
        nueva_cadena2 = ' '.join(cadena_2)
        df_nuevo.at[indice, "ClaimDescription"] = nueva_cadena2
    return df_nuevo

def Preprocesado(df):
    df = convert_tolower(df)
    df = clean_text(df)
    return df



def optimal_hierarchical_clustering(X):
    print("Hierarchical clustering exploration")
    max_silhouette = -1
    optimal_method = ''
    optimal_num_clusters = 0
    for method in ['single', 'complete', 'average', 'ward']:
        Z = linkage(X, method)
        for num_clusters in [2, 3, 4, 5, 6]:
            print(f"n_clusters: {num_clusters}, method: {method}")
            labels = fcluster(Z, num_clusters, criterion='maxclust')
            silhouette = silhouette_score(X, labels)
            if silhouette > max_silhouette:
                max_silhouette = silhouette
                optimal_method = method
                optimal_num_clusters = num_clusters
    print(f"Shilouette: {max_silhouette}")
    return optimal_method, optimal_num_clusters

def optimal_kmeans_clusters(X):
    print("Estudio clustering con K-Means")
    max_silhouette = -1
    optimal_num_clusters = 0
    optimal_n_init = 10
    optimal_init = ''

    for num_clusters in [2, 3, 4, 5]:
        for init_method in ['k-means++', 'random']:
            for n_init_value in [10, 20, 30]:
                print(f"n_clusters: {num_clusters}, init_method: {init_method}, n_init: {n_init_value}")
                kmeans = KMeans(n_clusters=num_clusters, init=init_method, n_init=n_init_value, random_state=42)
                labels = kmeans.fit_predict(X)
                silhouette = silhouette_score(X, labels)
                if silhouette > max_silhouette:
                    max_silhouette = silhouette
                    optimal_num_clusters = num_clusters
                    optimal_init = init_method
                    optimal_n_init = n_init_value
    print(f"Shilouette: {max_silhouette}")
    return optimal_num_clusters, optimal_init, optimal_n_init

def representar_clusters(X, labels):
    cmap='viridis'
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap=cmap)
    plt.show()

def optimal_gaussian_mixture(X):
    print("Estudio clustering con Gaussian Mixture")
    max_silhouette = -1
    optimal_num_clusters = 0
    optimal_covariance_type = ''
    optimal_n_init = 1
    optimal_labels = []
    for num_clusters in [2, 3, 4, 5]:
        for covariance_type in ['full', 'tied', 'diag', 'spherical']:
                for n_init_value in [1, 5, 10]:
                        print(f"Número de clusters: {num_clusters}, Covariance Type: {covariance_type}, n_init: {n_init_value}")
                        
                        gmm = GaussianMixture(n_components=num_clusters, covariance_type=covariance_type, n_init=n_init_value, random_state=42)
                        labels = gmm.fit_predict(X)
                        silhouette = silhouette_score(X, labels)
                        print(f"Coeficiente de silueta: {silhouette}")
                        
                        if silhouette > max_silhouette:
                            max_silhouette = silhouette
                            optimal_num_clusters = num_clusters
                            optimal_covariance_type = covariance_type
                            optimal_n_init = n_init_value
                            optimal_labels = labels
    
    print(f"Shilouette: {max_silhouette}")
    return optimal_num_clusters, optimal_covariance_type, optimal_n_init, optimal_labels


def optimal_dbscan_clusters(X): 
    print("DBScan clustering exploration")
    max_silhouette = -1
    optimal_num_clusters = 0
    optimal_eps = 0
    optimal_min_samples = 0
    for eps in [0.1, 0.4, 0.7]:
        for min_samples in [2, 5, 7]:
            print(f"Eps: {eps}, Min Samples: {min_samples}")
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            unique_labels = set(labels) - {-1}
            num_clusters = len(unique_labels)
            
            if num_clusters > 1:
                silhouette = silhouette_score(X, labels)
                if silhouette > max_silhouette:
                    max_silhouette = silhouette
                    optimal_num_clusters = num_clusters
                    optimal_eps = eps
                    optimal_min_samples = min_samples

    print(f"Shilouette: {max_silhouette}")

    return optimal_num_clusters, optimal_eps, optimal_min_samples



def extraer_caracteristicas(df):
    vectorizer = TfidfVectorizer(max_df=0.2, min_df=5,stop_words="english", norm="l1", tokenizer=word_tokenize)
    X = vectorizer.fit_transform(df['ClaimDescription'])
    X = (X - X.min()) / (X.max() - X.min())
    print(f"n_samples: {X.shape[0]}, n_features: {X.shape[1]}")
    return X.toarray()

def crear_PCA(X):
    pca = PCA(n_components=0.7)
    X_pca = pca.fit_transform(X)
    explained_variance_ratio = pca.explained_variance_ratio_
    explained_variance_ratio_cumsum = np.cumsum(explained_variance_ratio)
    num_components = pca.n_components_
    print(f"Número de componentes: {num_components}")
    print("Variance Ratio: ", explained_variance_ratio_cumsum[-1])

    #plt.figure(figsize=(10, 5))
    #plt.scatter(X_pca[:, 0], X_pca[:, 1], c=X_pca[:, 2], cmap='viridis', marker='o')
    #plt.title('Representación gráfica del PCA')
    #plt.colorbar(label='Tercera Dimensión')
    #plt.show()
    return X_pca, num_components

def calcular_matriz_similitud(datanorm, nombre_grafica):
    print("Matriz similitud")
    # 7. Obtención de Componentes Principales sobre el dataframe normalizado introducido y Similarity Matrix
    dist = sklearn.metrics.DistanceMetric.get_metric('euclidean')
    matsim = dist.pairwise(datanorm)
    return matsim

def analysis_process(df):
    X = extraer_caracteristicas(df)
    X_pca, num_components = crear_PCA(X)
    return X_pca, X


def plot_dbscan_clusters(X_pca, dbscan_labels):
    # Asumiendo que dbscan_labels es la salida de DBSCAN
    unique_labels = set(dbscan_labels)
    num_clusters = len(unique_labels)

    # Definir colores para cada cluster
    colors = sns.color_palette("husl", num_clusters)

    # Crear un diccionario para mapear labels a colores
    label_color_dict = {label: color for label, color in zip(unique_labels, colors)}

    # Asignar colores a cada punto basado en su label
    point_colors = [label_color_dict[label] for label in dbscan_labels]

    # Visualizar en el espacio PCA
    plt.figure(figsize=(10, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=point_colors, s=50)
    plt.title('DBSCAN Clusters')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.show()

def representaciónDBScan(X):
    print("Representacion DBScan")
    eps_optimal = 0.6
    min_samples_optimal = 6
    dbscan = DBSCAN(eps=eps_optimal, min_samples=min_samples_optimal)
    labels = dbscan.fit_predict(X)
    unique_labels = set(labels) - {-1}
    num_clusters = len(unique_labels)
    print(f"numero clusters: {num_clusters}")
    plot_dbscan_clusters(X, labels)



def clasificacion_NN(df):
    claim_descriptions = df['ClaimDescription'].values
    claim_costs = df['UltimateIncurredClaimCost'].values

    # Convertir las descripciones de reclamaciones en vectores TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000)
    x = vectorizer.fit_transform(claim_descriptions)

    # Normalizar los costos de las reclamaciones
    claim_costs = (claim_costs - claim_costs.min()) / (claim_costs.max() - claim_costs.min())

    # Concatenar los vectores TF-IDF y los costos de las reclamaciones
    x = np.concatenate([x.toarray(), claim_costs.reshape(-1, 1)], axis=1)

    # Definir el autoencoder
    input_layer = Input(shape=(x.shape[1],))
    encoded = Dense(500, activation='relu')(input_layer)
    encoded = Dense(500, activation='relu')(encoded)
    encoded = Dense(2000, activation='relu')(encoded)
    encoded = Dense(10, activation='relu')(encoded)
    decoded = Dense(2000, activation='relu')(encoded)
    decoded = Dense(500, activation='relu')(decoded)
    decoded = Dense(x.shape[1], activation='sigmoid')(decoded)  # Asegúrate de que la salida tenga la misma dimensión que la entrada

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Entrenar el autoencoder
    autoencoder.fit(x, x, epochs=10, batch_size=256, shuffle=True)

    # Usar el autoencoder para reducir la dimensionalidad de 'x'
    encoder = Model(input_layer, encoded)
    x = encoder.predict(x)

    # Aplicar K-means al resultado
    kmeans = KMeans(n_clusters=2)
    clusters = kmeans.fit_predict(x)
    return clusters, x

def count_labels(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    # Imprimir resultados
    print("Number of labels:", len(unique_labels))
    print("\nValues per label:")
    for label, count in zip(unique_labels, counts):
        print(f"label {label}: {count} values")

def main():
    df = cargar_train()
    df_preprocesado = Preprocesado(df)
    #df.to_csv("prueba.csv", sep=";", index=False)
    X_pca, X = analysis_process(df_preprocesado)
    #clustering_k_means(df['UltimateIncurredClaimCost'], X_pca, "k-means")
    #optimal_method, optimal_num_clusters = optimal_kmeans_clusters(X_pca) #0.22
    #print(f'El método de linkage óptimo es {optimal_method} con {optimal_num_clusters} clusters.')
    #optimal_num_clusters, optimal_eps, optimal_min_samples =  optimal_dbscan_clusters(X_pca)
    #print(f'Los clusters óptimos son {optimal_num_clusters} con {optimal_num_clusters} min_samples y {optimal_eps} eps')
    #representaciónDBScan(X_pca)
    #optimal_num_clusters, optimal_init, optimal_tol, optimal_n_init, optimal_max_iter, labels = optimal_kmeans_clusters(X_pca)
    #representar_clusters_kmeans(X, labels, optimal_num_clusters, optimal_init, optimal_tol, optimal_n_init, optimal_max_iter)
    #km = KMeans(n_clusters=2, init='random', n_init=20, max_iter=300, tol=0.0000001, random_state=42)
    #labels = km.fit_predict(X_pca)
    #representar_clusters_kmeans(X_pca, labels)
    #silhouette = silhouette_score(X_pca, labels)
    #print(silhouette)
    #optimal_num_clusters, optimal_init, optimal_tol, optimal_n_init, optimal_max_iter, labels = optimal_gaussian_mixture(X_pca)
    #gmm = GaussianMixture(n_components=num_clusters, covariance_type=covariance_type, tol=tol_value, n_init=n_init_value, max_iter=max_iter_value, random_state=42)
    #labels = gmm.fit_predict(X_pca)
    #representar_clusters_kmeans(X_pca, labels)
    """ cluster_labels, x = clasificacion_NN(df)
    df['ClusterLabels'] = cluster_labels
    df.to_csv("prueba_clusters.csv", sep=";", index=False)
    representar_clusters_kmeans(x, cluster_labels)
    silhouette = silhouette_score(x, cluster_labels)
    print(silhouette) """
    


    #method, num_clusters = optimal_hierarchical_clustering(X_pca)
    #print(method)
    #print(num_clusters) 
    kmeans = KMeans(n_clusters=2, init='k-means++', n_init=20, random_state=42)
    labels = kmeans.fit_predict(X_pca)
    df = pd.read_csv("datasets/train.csv")
    df['ClaimDescription_Cluster'] = labels
    df.to_csv('prueba_clusters.csv', sep=";", index=False)
    count_labels(labels)
    representar_clusters(X_pca, labels)



if __name__ == "__main__":
    main()

