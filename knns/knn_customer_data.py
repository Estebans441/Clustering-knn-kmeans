# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('segmented_customers.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print("----------------------")
print("Dataset")
print("----------------------")
print(X[:5])

# Preprocessing - encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

sc = StandardScaler()
X = sc.fit_transform(X)
print("----------------------")
print(X[:5])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)

# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print("----------------------")
print("Predicts")
print("----------------------")
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)[:5])

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
print("----------------------")
print("Confusion Matrix")
print("----------------------")
print(cm)
print("Accuracy: ", accuracy_score(y_test, y_pred))
