import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge

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

# RECEBER DADOS DE ENTRADA
Data = np.loadtxt("EMG.csv", delimiter=',')
N, p = Data.shape
colors = ['red', 'green', 'purple', 'blue', 'gray']

# DEFINIR ROTULOS DE CLASSIFICAÇÃO
neutro = np.tile(np.array([[1, -1, -1, -1, -1]]), (1000, 1))
sorrindo = np.tile(np.array([[-1, 1, -1, -1, -1]]), (1000, 1))
aberto = np.tile(np.array([[-1, -1, 1, -1, -1]]), (1000, 1))
surpreso = np.tile(np.array([[-1, -1, -1, 1, -1]]), (1000, 1))
rabugento = np.tile(np.array([[-1, -1, -1, -1, 1]]), (1000, 1))
Y = np.tile(np.concatenate((neutro, sorrindo, aberto, surpreso, rabugento)), (10, 1))

# NUMERO DE RODADAS
R = 100

# ARMANEZAR DADOS DE PRECISÃO
accuracies_lr = []  # PARA linear regression (MQO tradicional)
accuracies_knn = []  # PARA k-NN (classificador k vizinhos mais proximos)
accuracies_dmc = []  # PARA DMC (distancia minima do centroide)
accuracies_ridge = [] # PARA Ridge (MQO Regularizado)

for r in range(R):
    s = np.random.permutation(N)

    X = Data[s, :]
    Y = Y[s, :]

    X = np.concatenate((
        np.ones((N, 1)), X
    ), axis=1)

    # EMBARALHAR AS AMOSTRAS

    amostra_embaralhada = np.random.permutation(N)
    X_random = X[amostra_embaralhada,:]
    y_random = Y[amostra_embaralhada,:]

    # DIVIDIR TESTE E TREINO (EM X E Y) NA PROPORÇÃO 80-20

    X_treino = X_random[0:int(N*.8),:]  
    Y_treino = y_random[0:int(N*.8),:]  

    X_teste = X_random[int(N*.8):,:]
    Y_teste = y_random[int(N*.8):,:]

    # Linear Regression
    W_hat = np.linalg.pinv(X_treino.T @ X_treino) @ X_treino.T @ Y_treino


    Y_hat_lr = X_teste @ W_hat

    # ------------------------------

    # Calcular precisão para MQO Tradicional
    discriminante_lr = np.argmax(Y_hat_lr, axis=1)
    discriminante2 = np.argmax(Y_teste, axis=1)
    accuracy_lr = accuracy_score(discriminante2, discriminante_lr)
    accuracies_lr.append(accuracy_lr)

    # ------------------------------

    # k vizinhos mais proximos
    k = 5 
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Use only features, excluding the bias term
    knn.fit(X_treino[:, 1:], np.argmax(Y_treino, axis=1))

    Y_hat_knn = knn.predict(X_teste[:, 1:])

    # Calculando precisão para k vizinhos mais proximos
    accuracy_knn = accuracy_score(discriminante2, Y_hat_knn)
    accuracies_knn.append(accuracy_knn)

    # ------------------------------

    # DMC
    Y_hat_dmc = dmc_classifier(X_treino[:, 1:], np.argmax(Y_treino, axis=1), X_teste[:, 1:])

    # Calcular precisão para DMC
    accuracy_dmc = accuracy_score(discriminante2, Y_hat_dmc)
    accuracies_dmc.append(accuracy_dmc)

    # ------------------------------

    # MQO Regularizado
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_treino[:, 1:], Y_treino)

    Y_hat_ridge = ridge.predict(X_teste[:, 1:])

    # Calcular precisão para MQO Regularizado
    discriminante_ridge = np.argmax(Y_hat_ridge, axis=1)
    accuracy_ridge = accuracy_score(discriminante2, discriminante_ridge)
    accuracies_ridge.append(accuracy_ridge)

# Visualizar os resultados de precisão
plt.figure(figsize=(8, 6))
plt.plot(range(1, R + 1), accuracies_lr, label='MQO Tradicional', marker='o', linestyle='-', color='b')
plt.plot(range(1, R + 1), accuracies_knn, label='k-NN', marker='o', linestyle='-', color='r')
plt.plot(range(1, R + 1), accuracies_dmc, label='DMC', marker='o', linestyle='-', color='g')
plt.plot(range(1, R + 1), accuracies_ridge, label='MQO Regularizado', marker='o', linestyle='-', color='y')
plt.title('Precisão vs. Rodadas')
plt.xlabel('Rodadas')
plt.ylabel('Precisão')
plt.legend()
plt.grid(True)
plt.show()


