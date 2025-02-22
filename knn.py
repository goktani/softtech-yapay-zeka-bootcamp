# gerekli kutuphaneleri iceriye aktar
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# veri setini yukle
cancer = load_breast_cancer()
df = pd.DataFrame(data = cancer.data, columns = cancer.feature_names)
df["target"] = cancer.target

# feauture ve hedef degiskenleri elde et
X = cancer.data
y = cancer.target

# train test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=2)

# standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN tanimla ve train

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train) # training

# KNN test edelim ve dogruluk hesapla

y_pred = knn.predict(X_test)
y_pred_train = knn.predict(X_train)

accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy}")
print(f"Trainining Accuracy: {accuracy_score(y_train, y_pred_train)}")



















