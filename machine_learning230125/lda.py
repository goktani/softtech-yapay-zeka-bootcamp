from sklearn import datasets

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt

digits = datasets.load_digits()

X = digits.data
y = digits.target.astype(int)

lda = LDA(n_components=2)

x_lda = lda.fit_transform(X, y)

plt.figure()
plt.scatter(x_lda[:,0], x_lda[:,1], c = y, cmap = "tab10", alpha = 0.5)

# %% 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

iris = datasets.load_iris()

X= iris.data
y = iris.target
target_names = iris.target_names

pca = PCA(n_components=2)
X_r = pca.fit_transform(X)

lda = LDA(n_components=2)
X_r2 = lda.fit(X,y).transform(X)

plt.figure()
colors = ["red", "blue", "green"]

for color, i, target_name in zip(colors, [0,1,2], target_names):
    plt.scatter(X_r[y==i, 0], X_r[y==i,1], color = color, label = target_name)
plt.title("PCA")

plt.figure()
for color, i, target_name in zip(colors, [0,1,2], target_names):
    plt.scatter(X_r2[y==i, 0], X_r2[y==i,1], color = color, label = target_name)
plt.title("lda")






















