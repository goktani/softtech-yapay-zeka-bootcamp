import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(0)

X = 2 * np.random.rand(100,1) # 0-2 arasinda degerler olustur
y = 3 + 4*X + 1*np.random.rand(100,1)

plt.scatter(X,y)

lin_reg = LinearRegression()

lin_reg.fit(X,y)

plt.scatter(X,y, color = "black")
plt.plot(X, lin_reg.predict(X), color = "red", alpha = 0.8)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression")

# Y = Ax + B
# a: lin_reg.coef_[0][0]
# b: lin_reg.intercept_[0]
for i in range(100):
    plt.plot(X, lin_reg.coef_[0][0]*X + lin_reg.intercept_[0], color = "blue", alpha = 0.6)
plt.show()

# %% 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score

diabetes_X, diabetes_y = load_diabetes(return_X_y=True)

# y = ax + b
diabetes_X = diabetes_X[:, np.newaxis, 2]

diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

regr = LinearRegression()

regr.fit(diabetes_X_train, diabetes_y_train)

diabetes_y_pred = regr.predict(diabetes_X_test)

rmse = np.sqrt(mean_squared_error(diabetes_y_pred, diabetes_y_test))
print(f"rmse: {rmse}")

r2_score_ = r2_score(diabetes_y_test, diabetes_y_pred)
print(f"r2_score: {r2_score_}")




























