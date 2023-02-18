# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('segmented_customers.csv')
X = dataset.iloc[:, 2:-1].values
y = dataset.iloc[:, -1].values
print("----------------------")
print("Dataset")
print("----------------------")
print(X[:5])

# Preprocessing - encoding categorical data
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)
print("----------------------")
print(X[:5])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)

# Defining the best number of neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

wcss = []
for i in range(1, 9):
    classifier = KNeighborsClassifier(n_neighbors=i, metric='minkowski', p=2)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Cm ", i)
    print(cm)
    print(accuracy_score(y_test, y_pred))
    wcss.append(accuracy_score(y_test, y_pred) * 100)
plt.plot(range(1, 9), wcss)
plt.title('Accuracy score using k neighbors')
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy score')
plt.show()

# Training the K-NN model on the Training set
classifier = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)
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

# Defining colors for each category
colors = ['red', 'blue', 'yellowgreen', 'orange', 'gold', 'mediumorchid']

# Visualising the Training set and Test set results
X_set_train, y_set_train = sc.inverse_transform(X_train), y_train
X_set_test, y_set_test = sc.inverse_transform(X_test), y_test

# Age vs Spending Score
for i, X_set, y_set in [(1, X_set_train, y_set_train), (0.36, X_set_test, y_set_test)]:
    for j in range(6):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 2],
                    color=colors[j], alpha=i, label='Cluster {}'.format(j))
plt.title("KNN clustering")
plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.legend()
plt.show()

# Age vs Annual Income
for i, X_set, y_set in [(1, X_set_train, y_set_train), (0.36, X_set_test, y_set_test)]:
    for j in range(6):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    color=colors[j], alpha=i, label='Cluster {}'.format(j))
plt.title("KNN clustering")
plt.xlabel("Age")
plt.ylabel("Annual Income")
plt.legend()
plt.show()

# Age vs Annual Income
for i, X_set, y_set in [(1, X_set_train, y_set_train), (0.36, X_set_test, y_set_test)]:
    for j in range(6):
        plt.scatter(X_set[y_set == j, 1], X_set[y_set == j, 2],
                    color=colors[j], alpha=i, label='Cluster {}'.format(j))
plt.title("KNN clustering")
plt.ylabel("Spending Score")
plt.xlabel("Annual Income")
plt.legend()
plt.show()

# Visualising the Dataset in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i, X_set, y_set in [(1, sc.inverse_transform(X_train), y_train), (0.36, sc.inverse_transform(X_test), y_test)]:
    for j in range(6):
        alpha_val = i if j in y_set else 0
        ax.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], X_set[y_set == j, 2],
                   color=colors[j], alpha=alpha_val, label='Cluster {}'.format(j))
ax.set_title("KNN clustering")
ax.set_xlabel("Age")
ax.set_ylabel("Annual Income (k$)")
ax.set_zlabel("Spending Score (1-100)")
plt.show()

# Interactive 3D plot
import plotly.express as px

df_train = pd.DataFrame(data=X_train, columns=["Age", "Annual Income", "Spending Score"])
df_train["Target"] = y_train
df_test = pd.DataFrame(data=X_test, columns=["Age", "Annual Income", "Spending Score"])
df_test["Target"] = y_test
fig = px.scatter_3d(df_train, x="Age", y="Annual Income", z="Spending Score", color="Target", opacity=1,
                    title="KNN clustering")
fig.add_trace(
    px.scatter_3d(df_test, x="Age", y="Annual Income", z="Spending Score", color="Target", opacity=0.6).data[0])
fig.show()
