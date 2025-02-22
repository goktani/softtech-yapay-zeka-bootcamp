from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

# Veri setini yükle
iris = load_iris()

# Veri setini eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Naive Bayes sınıflandırıcıyı oluştur
nb_clf = GaussianNB()

# Modeli eğit
nb_clf.fit(X_train, y_train)

# Test veri seti üzerinde modelin performansını değerlendir
y_pred = nb_clf.predict(X_test)

# Sınıflandırma raporunu göster
print(classification_report(y_test, y_pred, target_names=iris.target_names))