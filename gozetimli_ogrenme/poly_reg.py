import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Rastgele non-lineer veri oluştur
X = np.random.rand(100, 1) * 4  # [0, 4] arasında rastgele x değerleri
y = 2 + 3 * X**2 + np.random.randn(100, 1)  # Gerçek bir non-lineer ilişkiyi taklit eden y değerleri (2 + 3x^2 + gürültü)

# Veriyi ve modeli görselleştir
plt.scatter(X, y)

# Polinom özelliklerini oluştur (2. dereceden polinom)
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)

# Polinom regresyon modelini oluştur
poly_reg = LinearRegression()

# Modeli eğit
poly_reg.fit(X_poly, y)

# Polinom regresyon modelinin tahmin ettiği çizgi
X_fit = np.linspace(0, 4, 100).reshape(-1, 1)
X_fit_poly = poly_features.transform(X_fit)
y_fit = poly_reg.predict(X_fit_poly)
plt.plot(X_fit, y_fit, color='red', label='Polinom Regresyon')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polinom Regresyon Modeli')
plt.legend()
plt.show()
