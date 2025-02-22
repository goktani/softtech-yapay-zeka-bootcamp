from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

plt.figure()
for i in range(len(iris.target_names)):
    plt.scatter(X_pca[y == i, 0], X_pca[y ==i,1], label = iris.target_names[i])
    
# %%
import mpl_toolkits.mplot3d
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

fig = plt.figure(1, figsize=(8,6))
ax = fig.add_subplot(111, projection = "3d", elev = -150, azim = 110)
X_reduced = PCA(n_components=3).fit_transform(X)
ax.scatter(
    X_reduced[:,0],
    X_reduced[:,1],
    X_reduced[:,2],
    c = iris.target,
    s = 40)













