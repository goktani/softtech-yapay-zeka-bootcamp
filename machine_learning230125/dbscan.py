import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.cluster import DBSCAN

X, _ = make_circles(n_samples = 1000, factor = 0.5, noise = 0.05, random_state = 42)

plt.figure()
plt.scatter(X[:,0], X[:,1])

dbscan = DBSCAN(eps = 0.1, min_samples=5)
cluster_labels = dbscan.fit_predict(X)

plt.figure()
plt.scatter(X[:,0], X[:,1], c = cluster_labels )

# %% 
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN 

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN 

# Yay şeklinde veri oluşturma
X, _ = make_moons(n_samples=1000, noise=0.05, random_state=42)

# Orijinal veriyi görselleştir
plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.title("Orijinal Veri")

# DBSCAN ile kümeleme
dbscan = DBSCAN(eps=0.2, min_samples=10)
cluster_labels = dbscan.fit_predict(X)

# Kümeleme sonuçlarını görselleştir
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis')
plt.title("DBSCAN ile Kümeleme")
plt.colorbar(label="Küme Etiketleri")
plt.show()