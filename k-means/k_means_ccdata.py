# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('CC GENERAL.csv')
dataset.drop(['CUST_ID'], axis=1, inplace=True)
dataset.dropna(subset=['CREDIT_LIMIT'], inplace=True)
dataset['MINIMUM_PAYMENTS'].fillna(dataset['MINIMUM_PAYMENTS'].median(), inplace=True)
X1 = dataset.iloc[:, :].values

# Preprocessing de data
cols = ['BALANCE', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'ONEOFF_PURCHASES_FREQUENCY',
        'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',
        'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT']
for col in cols:
    dataset[col] = np.log(1 + dataset[col])

# Decomposition
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)
X = pca.fit_transform(dataset)

# Extracting the data
print(dataset.head())
X = dataset.iloc[:, :].values

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=23)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=23)
y_kmeans = kmeans.fit_predict(X)
# Visualising the clusters

# ONE_OFF_PURCHASES VS PURCHASES
plt.scatter(X1[y_kmeans == 0, 3], X1[y_kmeans == 0, 2], s=5, c='blue', label='Cluster 1')
plt.scatter(X1[y_kmeans == 1, 3], X1[y_kmeans == 1, 2], s=5, c='orange', label='Cluster 2')
plt.scatter(X1[y_kmeans == 2, 3], X1[y_kmeans == 2, 2], s=5, c='green', label='Cluster 3')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=50, c='red', label='Centroids')
plt.title('Distribution of clusters (3) based on ONE_OFF_PURCHASES and PURCHASES')
plt.xlabel('ONE_OFF_PURCHASES')
plt.ylabel('PURCHASES')
plt.legend()
plt.show()

# CREDIT LIMIT VS PURCHASES
plt.scatter(X1[y_kmeans == 0, 12], X1[y_kmeans == 0, 2], s=5, c='blue', label='Cluster 1')
plt.scatter(X1[y_kmeans == 1, 12], X1[y_kmeans == 1, 2], s=5, c='orange', label='Cluster 2')
plt.scatter(X1[y_kmeans == 2, 12], X1[y_kmeans == 2, 2], s=5, c='green', label='Cluster 3')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=50, c='red', label='Centroids')
plt.title('Distribution of clusters (3) based on CREDIT_LIMIT and PURCHASES')
plt.xlabel('CREDIT_LIMIT')
plt.ylabel('PURCHASES')
plt.legend()
plt.show()
