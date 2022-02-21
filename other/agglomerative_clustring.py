from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import homogeneity_score
import numpy as np

iris = datasets.load_iris()
X=iris.data
y=iris.target

Cluster = AgglomerativeClustering(n_clusters=3,linkage="ward")
pred_y=Cluster.fit_predict(X)

plt.scatter(X[pred_y == 0,0],X[pred_y == 0,1],s=100,c="red",label="Iris-Sentosa")
plt.scatter(X[pred_y == 1,0],X[pred_y == 1,1],s=100,c="blue",label="Iris-Versicolour")
plt.scatter(X[pred_y == 2,0],X[pred_y == 2,1],s=100,c="green",label="Iris-Verginica")

plt.legend()
plt.show()
