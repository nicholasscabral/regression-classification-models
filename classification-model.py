import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Define the DMC classifier function
def dmc_classifier(X_train, Y_train, X_test):
    centroids = []
    for label in np.unique(Y_train):
        class_samples = X_train[Y_train == label]
        centroid = np.mean(class_samples, axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)

    Y_pred = np.argmin(np.linalg.norm(X_test[:, np.newaxis] - centroids, axis=2), axis=1)
    return Y_pred

# Load data from CSV
Data = np.loadtxt("EMG.csv", delimiter=',')
N, p = Data.shape
colors = ['red', 'green', 'purple', 'blue', 'gray']

# Create class labels
neutro = np.tile(np.array([[1, -1, -1, -1, -1]]), (1000, 1))
sorrindo = np.tile(np.array([[-1, 1, -1, -1, -1]]), (1000, 1))
aberto = np.tile(np.array([[-1, -1, 1, -1, -1]]), (1000, 1))
surpreso = np.tile(np.array([[-1, -1, -1, 1, -1]]), (1000, 1))
rabugento = np.tile(np.array([[-1, -1, -1, -1, 1]]), (1000, 1))
Y = np.tile(np.concatenate((neutro, sorrindo, aberto, surpreso, rabugento)), (10, 1))

# Number of iterations
R = 100

# Store accuracy values
accuracies_lr = []  # For linear regression
accuracies_knn = []  # For k-NN
accuracies_dmc = []  # For DMC

for r in range(R):
    s = np.random.permutation(N)

    X = Data[s, :]
    Y = Y[s, :]

    X = np.concatenate((
        np.ones((N, 1)), X
    ), axis=1)

    # Split the data into training and testing sets
    split_ratio = 0.8  # 80% for training, 20% for testing
    split_index = int(N * split_ratio)

    X_treino = X[:split_index, :]
    Y_treino = Y[:split_index, :]

    X_teste = X[split_index:, :]
    Y_teste = Y[split_index:, :]

    # Linear Regression
    lb = 0.1
    W_hat = np.linalg.pinv(X_treino.T @ X_treino) @ X_treino.T @ Y_treino
    

    Y_hat_lr = X_teste @ W_hat

    # Calculate accuracy for linear regression
    discriminante_lr = np.argmax(Y_hat_lr, axis=1)
    discriminante2 = np.argmax(Y_teste, axis=1)

    accuracy_lr = accuracy_score(discriminante2, discriminante_lr)
    accuracies_lr.append(accuracy_lr)

    # k-Nearest Neighbors
    k = 5  # You can change the value of k as needed
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Use only features, excluding the bias term
    knn.fit(X_treino[:, 1:], np.argmax(Y_treino, axis=1))

    Y_hat_knn = knn.predict(X_teste[:, 1:])

    # Calculate accuracy for k-NN
    accuracy_knn = accuracy_score(discriminante2, Y_hat_knn)
    accuracies_knn.append(accuracy_knn)

    # DMC
    Y_hat_dmc = dmc_classifier(X_treino[:, 1:], np.argmax(Y_treino, axis=1), X_teste[:, 1:])

    # Calculate accuracy for DMC
    accuracy_dmc = accuracy_score(discriminante2, Y_hat_dmc)
    accuracies_dmc.append(accuracy_dmc)

# Visualize the results (accuracy)
plt.figure(figsize=(8, 6))
plt.plot(range(1, R + 1), accuracies_lr, label='Linear Regression', marker='o', linestyle='-', color='b')
plt.plot(range(1, R + 1), accuracies_knn, label='k-NN', marker='o', linestyle='-', color='r')
plt.plot(range(1, R + 1), accuracies_dmc, label='DMC', marker='o', linestyle='-', color='g')
plt.title('Accuracy vs. Iteration')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
