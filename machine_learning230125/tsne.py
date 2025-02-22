from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

digits = datasets.load_digits()

X = digits.data
y = digits.target.astype(int)

tsne = TSNE(n_components=2)

x_tsne  = tsne.fit_transform(X)

plt.figure()
plt.scatter(x_tsne[:,0], x_tsne[:,1], c = y, cmap = "tab10", alpha = 0.5)

"""
ödev: obezite veri seti ile sınıflandirma calismasi
https://www.kaggle.com/datasets/ruchikakumbhar/obesity-prediction/code?datasetId=6479256&sortBy=voteCount

1) 3 farklı siniflandirma algoritmasını kullanarak bir sınıfladırma yapalım
sonucları classification report ile degerlendirme ve karsilastirma

2) pca analizi uygula ve 2 component'i elde ederek bir siniflandiriciya sokalim
sonucları classification report ile degerlendirme

"""