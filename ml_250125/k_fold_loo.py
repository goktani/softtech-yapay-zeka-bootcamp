from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut

# Iris veri setini yükle
iris = load_iris()
X = iris.data
y = iris.target

# Veri setini eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar Ağacı modelini tanımla
tree = DecisionTreeClassifier()

# Karar Ağacı için hyperparameter aralıklarını belirle
tree_param_dist = {'max_depth': [3, 5, 7],
                   'max_leaf_nodes': [None, 5, 10, 20, 30, 50]}

kf = KFold(n_splits=5)
tree_grid_search_kf = GridSearchCV(tree, tree_param_dist, cv = kf)
tree_grid_search_kf.fit(X_train, y_train)
print(f"Decision tree grid search kfold en iyi sonuc: {tree_grid_search_kf.best_params_}")
print(f"Decision tree grid search kfold en iyi sonuc: {tree_grid_search_kf.best_score_}")

loo = LeaveOneOut()
tree_grid_search_loo = GridSearchCV(tree, tree_param_dist, cv = loo)
tree_grid_search_loo.fit(X_train, y_train)
print(f"Decision tree grid search loo en iyi sonuc: {tree_grid_search_loo.best_params_}")
print(f"Decision tree grid search loo en iyi sonuc: {tree_grid_search_loo.best_score_}")


#knn tnaımlama

knn = KNeighborsClassifier()
knn_param_grid= {"n_neighbors": np.arange(2,30)}

kf = KFold(n_splits=5)
knn_grid_search = GridSearchCV(knn,knn_param_grid, cv=kf)
knn_grid_search.fit(X_train, y_train)

print(knn_grid_search.best_params_)
print(knn_grid_search.best_score_)


lvo = LeaveOneOut()
knn_grid_search_lvo = GridSearchCV(knn,knn_param_grid, cv=lvo)
knn_grid_search_lvo.fit(X_train, y_train)

print(knn_grid_search_lvo.best_params_)
print(knn_grid_search_lvo.best_score_)

