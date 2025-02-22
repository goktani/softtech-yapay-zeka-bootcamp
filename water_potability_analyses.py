import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import missingno as msno
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import precision_score, confusion_matrix
from sklearn import tree

df = pd.read_csv("water_potability.csv")
# print(df.head())
# describe
# print(df.describe())
# print(df.info())

# Dependent Variable Analysis
d = df["Potability"].value_counts().reset_index()
d.columns = ["Potability", "Count"]
fig = px.pie(d, values="Count", names=["Not Potable", "Potable"], hole=0.35, opacity=0.8,
             labels={"label": "Potability", "Potability": "Number of Samples"})
fig.update_layout(title=dict(text="Pie Chart of Potability Feature"))
fig.update_traces(textposition="outside", textinfo="percent+label")
# fig.show()

# Correlation Between Features
# print(df.corr())
sns.clustermap(df.corr(), cmap="vlag", dendrogram_ratio=(0.1, 0.2), annot=True, linewidths=.8, figsize=(9,10))
# plt.show()

# Distribution of Features
non_potable = df.query("Potability == 0")
potable = df.query("Potability == 1")

plt.figure(figsize=(15,15))
for ax, col in enumerate(df.columns[:9]):
    plt.subplot(3,3, ax + 1)
    plt.title(col)
    sns.kdeplot(x=non_potable[col], label="Non Potable")
    sns.kdeplot(x=potable[col], label="Potable")
    plt.legend()
plt.tight_layout()
# plt.show()

# Preprocessing: Missing Value Problem
msno.matrix(df)
# plt.show()

# print(df.isnull().sum())

# handle missing value with average of features
df["ph"] = df["ph"].fillna(df["ph"].mean())
df["Sulfate"] = df["Sulfate"].fillna(df["Sulfate"].mean())
df["Trihalomethanes"] = df["Trihalomethanes"].fillna(df["Trihalomethanes"].mean())

# print(df.isnull().sum())

# Preprocessing: Train-Test Split and Normalization
X = df.drop("Potability", axis=1).values
y = df["Potability"].values

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
# print("X_train",X_train.shape)
# print("X_test",X_test.shape)
# print("y_train",y_train.shape)
# print("y_test",y_test.shape)

# min-max normalization
x_train_max = np.max(X_train)
x_train_min = np.min(X_train)
X_train = (X_train - x_train_min) / (x_train_max - x_train_min)
X_test = (X_test - x_train_min) / (x_train_max - x_train_min)

# Modelling: Decision Tree and Random Forest Classifiers
# Precision Score: The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives.
# The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.

models = [("DTC", DecisionTreeClassifier(max_depth=3)),
          ("RF", RandomForestClassifier())]
finalResults = []
cmList = []
for name, model in models:
    model.fit(X_train, y_train)  # train
    model_result = model.predict(X_test)  # prediction
    score = precision_score(y_test, model_result)
    cm = confusion_matrix(y_test, model_result)

    finalResults.append((name, score))
    cmList.append((name, cm))
# print(finalResults)
for name, i in cmList:
    plt.figure()
    sns.heatmap(i, annot=True, linewidths=0.8, fmt=".1f")
    plt.title(name)
    # plt.show()

# Visualize Decision Tree
dt_clf = models[0][1]
# print(dt_clf)

plt.figure(figsize=(25,20))
tree.plot_tree(dt_clf,
               feature_names=df.columns.tolist()[:-1],
               class_names=["0", "1"],
               filled=True,
               precision=5)
# plt.show()

# Random Forest Hyperparameter Tuning
model_params = {
    "Random Forest": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": [10, 50, 100],
            "max_features": ["sqrt", "log2"],  # Removed 'auto'
            "max_depth": list(range(1, 21, 3))
        }
    }
}
# print(model_params)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2)
scores = []
for model_name, params in model_params.items():
    rs = RandomizedSearchCV(params["model"], params["params"], cv=cv, n_iter=10)
    rs.fit(X, y)
    scores.append([model_name, dict(rs.best_params_), rs.best_score_])
print(scores)