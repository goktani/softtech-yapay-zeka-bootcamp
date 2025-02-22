import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Veri setini yükleme
data = pd.read_csv("Obesity_prediction.csv")

# Veri setini ekrana yazdır (İlk bakış için)
print(data)
print(data.columns)  # Sütun isimlerini yazdır

# Kategorik değişkenleri belirleme
categorical_columns = ['Gender', 'family_history', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

# Kategorik değişkenleri sayısal hale getirme (Ordinal Encoding kullanarak)
encoder = OrdinalEncoder()
data[categorical_columns] = encoder.fit_transform(data[categorical_columns])

# Bağımsız değişkenler (X) ve bağımlı değişken (y) ayırma
X = data.drop('Obesity', axis=1)  # 'Obesity' sütunu hedef değişken olduğu için çıkarılıyor
y = data['Obesity']  # Hedef değişken

# Veriyi eğitim ve test setlerine ayırma (Eğitim için %80, test için %20 kullanılıyor)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Kullanılacak sınıflandırma algoritmaları
algorithms = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "KNN": KNeighborsClassifier()
}

# Sonuçları saklamak için boş bir sözlük
results = {}

# Algoritmaların eğitimi ve test edilmesi
for name, model in algorithms.items():
    model.fit(X_train, y_train)  # Modeli eğit
    y_pred = model.predict(X_test)  # Test verisi üzerinde tahmin yap
    results[name] = classification_report(y_test, y_pred, output_dict=True)  # Sınıflandırma raporunu sakla

# Sonuçları ekrana yazdırma
for name, report in results.items():
    print(f"\n{name} Classification Report:\n")
    print(classification_report(y_test, model.predict(X_test)))

# PCA ile Boyut İndirgeme
# Önce veriyi ölçeklendiriyoruz, çünkü PCA ölçeklendirilmiş verilere daha iyi çalışır
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA ile boyutu 2'ye indir
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# PCA ile indirgenmiş veriyi eğitim ve test setlerine ayırma
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# PCA ile indirgenmiş veri üzerinde sınıflandırma yapmak için model oluşturma
pca_model = RandomForestClassifier(random_state=42)
pca_model.fit(X_train_pca, y_train_pca)  # Modeli eğit
y_pred_pca = pca_model.predict(X_test_pca)  # Test verisi üzerinde tahmin yap

# PCA sonrası sınıflandırma performansını değerlendirme
pca_report = classification_report(y_test_pca, y_pred_pca)
print("\nPCA + Random Forest Classification Report:\n")
print(pca_report)
