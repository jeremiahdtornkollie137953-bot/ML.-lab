import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import silhouette_score
     

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
     

# Perform k-means clustering
k = 3  # number of clusters
kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
     

# Get cluster labels for k-means
labels = kmeans.labels_
     

#print silhouette score for kmeans
print("Silhouette score for k-means is: ", (silhouette_score(X, labels, metric='euclidean')))

# Perform Agglomerative clustering (as alternative to k-medoids)
k = 3  # number of clusters
agglom = AgglomerativeClustering(n_clusters=k).fit(X)
     
# Get cluster labels for agglomerative clustering
labels = agglom.labels_

#print silhouette score for agglomerative clustering
print("Silhouette score for agglomerative clustering is: ", (silhouette_score(X, labels, metric='euclidean')))
     
# Visualize the clusters (using first two features) for k-means
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='Green', s=200, alpha=0.75)
plt.title('KMeans Clustering of Iris Dataset')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()