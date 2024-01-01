import nltk
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

stopwords = set(stopwords.words("english"))
porter=nltk.PorterStemmer()

import pandas as pd
import numpy as np

import unidecode
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from sklearn.metrics import silhouette_score
def cargar_train():
    df = pd.read_csv("Task 2/preprocesing.csv")
    columnas_necesarias = ['UltimateIncurredClaimCost','Claim_Body_Parts','Claim_Injuries']
    todas_las_columnas = df.columns.tolist()
    columnas_a_eliminar = [col for col in todas_las_columnas if col not in columnas_necesarias]
    df = df.drop(columns=columnas_a_eliminar)
    return df


def Preprocesado():
    pass

def crear_PCA(X):
    pca = PCA(n_components=0.7)
    X_pca = pca.fit_transform(X)
    explained_variance_ratio = pca.explained_variance_ratio_
    explained_variance_ratio_cumsum = np.cumsum(explained_variance_ratio)
    num_components = pca.n_components_
    print(f"n_features: {num_components}")
    print("Variance Ratio: ", explained_variance_ratio_cumsum[-1])

    plt.figure(figsize=(10, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=X_pca[:, 2], cmap='viridis', marker='o')
    plt.title('PCA')
    plt.colorbar(label='Third dimension')
    plt.show()
    return X_pca


def extraer_caracteristicas(df, columna):
    """max_df=0.8,min_df=5,""" 
    vectorizer = TfidfVectorizer(stop_words="english", norm="l1", tokenizer=word_tokenize)
    X = vectorizer.fit_transform(df[columna])
    print(f"n_samples: {X.shape[0]}, n_features: {X.shape[1]}")
    return X.toarray()

def representar_clusters(X, labels):
    cmap='viridis'
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap=cmap)
    plt.show()

def count_labels(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    # Imprimir resultados
    print("Number of labels:", len(unique_labels))
    print("\nValues per label:")
    for label, count in zip(unique_labels, counts):
        print(f"label {label}: {count} values")

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

def optimal_gaussian_mixture(X):
    print("Estudio clustering con Gaussian Mixture")
    max_silhouette = -1
    optimal_num_clusters = 0
    optimal_covariance_type = ''
    optimal_n_init = 1
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

    print(f"Shilouette: {max_silhouette}")
    return optimal_num_clusters, optimal_covariance_type, optimal_n_init

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

def main():
    df = cargar_train()
    Preprocesado()
    #columna = 'Claim_Body_Parts' # 0.31 2 random 10
    columna = 'Claim_Injuries'    # 0.65 5 k-means++
    X = extraer_caracteristicas(df, columna)
    X_pca = crear_PCA(X)
    optimal_num_clusters, optimal_init, optimal_n_init = optimal_kmeans_clusters(X_pca)
    print(optimal_num_clusters)
    print(optimal_init)
    print(optimal_n_init)

if __name__ == "__main__":
    main()