import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

X, _ = make_blobs(n_samples=3000, centers=10, cluster_std=0.2, random_state=0)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], s = 50, alpha=0.7, edgecolors="k")
plt.title("Örnek veri dagilimi")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

model = AgglomerativeClustering(n_clusters=10)
cluster_labels = model.fit_predict(X)

plt.figure()
dendrogram(linkage(X, method = "ward"), no_labels = True)

