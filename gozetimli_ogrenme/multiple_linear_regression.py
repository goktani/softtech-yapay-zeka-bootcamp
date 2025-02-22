import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(0)

X = np.random.rand(100,2)
coefficient = np.array([3,5])
y = np.dot(X, coefficient) + np.random.rand(100)

# y = a0 + a1*x1 + a2*x2 = a0 + 3*x1 + 5*x2
fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")
ax.scatter(X[:,0], X[:,1], y, c = "b", marker = "o")
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("y")

lin_reg = LinearRegression()

lin_reg.fit(X,y)

print(f"coefficient: {lin_reg.coef_}")
print(f"intercept: {lin_reg.intercept_}")

fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")
ax.scatter(X[:,0], X[:,1], y, c = "b", marker = "o")
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("y")

# modeli gorsellestir
x1, x2 = np.meshgrid(np.linspace(0,1,10), np.linspace(0, 1, 10))
y_pred = lin_reg.predict(np.array([x1.flatten(), x2.flatten()]).T)
ax.plot_surface(x1,x2,y_pred.reshape(x1.shape), alpha = 0.5)

plt.title()
plt.show()

# %%
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Diabetes veri setini yükle
diabetes = load_diabetes()

# Bağımsız değişkenleri ve hedef değişkeni ayır
X = diabetes.data
y = diabetes.target

# Veri setini eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lineer regresyon modelini oluştur
lin_reg = LinearRegression()

# Modeli eğit
lin_reg.fit(X_train, y_train)

# Eğitim ve test veri setleri üzerinde modelin performansını değerlendir
test_pred = lin_reg.predict(X_test)

test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

print("Test seti RMSE:", test_rmse)
































