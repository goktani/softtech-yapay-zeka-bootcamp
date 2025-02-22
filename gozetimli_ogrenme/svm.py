# import libraries
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

digits = load_digits()
plt.figure(figsize=(10,5))
for i in range(2):
    plt.subplot(1,2,i+1)
    plt.imshow(digits.images[i], cmap="gray")
    plt.title(f"görüntü{i+1}")
    plt.axis("off")
plt.show()

X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

clf = svm.SVC()
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(f"CLF Accuracy:{accuracy}")