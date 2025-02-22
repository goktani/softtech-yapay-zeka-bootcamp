from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# iris veri setini yukle
iris = load_iris()

df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
df["target"] = iris.target

X = iris.data
y = iris.target 

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=42)

tree_clf = DecisionTreeClassifier(criterion="gini", max_depth=30, random_state=42)
tree_clf.fit(X_train, y_train)


y_pred = tree_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"test accuracy: {accuracy}")

y_pred_train = tree_clf.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
print(f"train accuracy: {accuracy_train}")

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot = True, cmap = "Blues", fmt = "g", xticklabels=iris.target_names)
plt.xlabel("tahmin edilen degerler")
plt.ylabel("gercek sinif")
plt.title("confusion matrix")

"""
test accuracy: 1.0
train accuracy: 0.9916666666666667

Ezber örneği (overfitting):
    train accuracy: 0.99
    test accuracy: 0.8    
"""

plt.figure(figsize=(15,10))
plot_tree(tree_clf, filled = True, feature_names=iris.feature_names, class_names=list(iris.target_names))

feature_importances = tree_clf.feature_importances_

feature_names = iris.feature_names

feature_importance_sorted = sorted(zip(feature_importances, feature_names), reverse=True)

for importance, feature_name in feature_importance_sorted:
    print(f"{feature_name}: {importance}")






