from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
tree = DecisionTreeClassifier()

tree_param_dist = {"max_depth":[3,5,7],
                   "max_leaf_nodes":[None,5,10,20,30,50]}

tree_grid_search_cv = GridSearchCV(tree, tree_param_dist, cv = 5)
tree_grid_search_cv.fit(X_train,y_train)
print(f"Decision tree grid search (5-fold) en iyi sonuc: {tree_grid_search_cv.best_params_}")
print(f"Decision tree grid search (5-fold) en iyi skor: {tree_grid_search_cv.best_score_}")

for mean_score, params in zip(tree_grid_search_cv.cv_results_["mean_test_score"], tree_grid_search_cv.cv_results_["params"]):
    print(f"ortalama test skoru: {mean_score} parametreler: {params}")
    
cv_results = tree_grid_search_cv.cv_results_
for i, params in enumerate(cv_results["params"]):
    print(f"parameters: {params}")
    for j in range(5):
        accuracy = cv_results[f"split{j}_test_score"][i]
        print(f"Fold {j+1} - accuracy: {accuracy:.4f}")    
    
    
    
    
    
    
    
    
    
    
    
    
    

