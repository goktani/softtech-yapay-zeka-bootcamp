import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples = 300, centers = 4, cluster_std = 0.5, random_state=0)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], s = 50, alpha=0.7, edgecolors="k")
plt.title("Örnek veri dagilimi")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

labels = kmeans.labels_

plt.figure()
plt.scatter(X[:, 0], X[:,1], c = labels, cmap="viridis", s = 50 , alpha = 0.7, edgecolors="k")

centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c = "red", s = 200, marker = "X")
