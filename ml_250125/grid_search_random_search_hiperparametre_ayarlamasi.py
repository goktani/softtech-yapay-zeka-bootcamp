"""
grid search ve random search
iris veri seti
knn, dt, svm
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier()
knn_param_grid = {"n_neighbors":np.arange(2,30)}

knn_grid_search = GridSearchCV(knn, knn_param_grid)    
knn_grid_search.fit(X_train, y_train)
print(f"KNN Grid search icin en iyi sonuclar: {knn_grid_search.best_params_}")
print(f"KNN Grid search en iyi skor: {knn_grid_search.best_score_}")

knn_random_search = RandomizedSearchCV(knn, knn_param_grid, n_iter = 10)
knn_random_search.fit(X_train, y_train)
print(f"KNN Random search icin en iyi sonuclar: {knn_random_search.best_params_}")
print(f"KNN Random search en iyi skor: {knn_random_search.best_score_}")

tree = DecisionTreeClassifier()
tree_param_grid = {"max_depth":[3,5,7],
                   "max_leaf_nodes":[None,5,10,20,30,50]}

tree_grid_search = GridSearchCV(tree, tree_param_grid)
tree_grid_search.fit(X_train, y_train)
print(f"DT Grid search icin en iyi sonuclar: {tree_grid_search.best_params_}")
print(f"DT Grid search en iyi skor: {tree_grid_search.best_score_}")

tree_random_search = RandomizedSearchCV(tree, tree_param_grid, n_iter = 18)
tree_random_search.fit(X_train, y_train)
print(f"DT Random search icin en iyi sonuclar: {tree_random_search.best_params_}")
print(f"DT Random search en iyi skor: {tree_random_search.best_score_}")

# SVM
svm = SVC()
svm_grid_param = {"C":[0.1,1,10,100],
                  "gamma":[0.1,0.01,0.001,0.0001]}
svm_grid_search = GridSearchCV(svm, svm_grid_param)
svm_grid_search.fit(X_train, y_train)
print(f"SVM Grid search icin en iyi sonuclar: {svm_grid_search.best_params_}")
print(f"SVM Grid search en iyi skor: {svm_grid_search.best_score_}")

svm_random_search = RandomizedSearchCV(svm, svm_grid_param)
svm_random_search.fit(X_train, y_train)
print(f"SVM random search icin en iyi sonuclar: {svm_random_search.best_params_}")
print(f"SVM random search en iyi skor: {svm_random_search.best_score_}")








