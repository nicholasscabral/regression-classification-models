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
R = 10

# ARMANEZAR DADOS DE PRECISÃO
accuracies_lr = []  # PARA linear regression (MQO tradicional)
accuracies_knn = []  # PARA k-NN (classificador k vizinhos mais proximos)
accuracies_dmc = []  # PARA DMC (distancia minima do centroide)
accuracies_ridge = [] # PARA Ridge (MQO Regularizado)

alphasValores = np.arange(0.1, 1.0, 0.1) # Logaritimo pequeno para amostra pequena, precisamos de um range pequeno para maior precisão do modelo

# Armazena acuracia para cada alpha (Apenas para o MQO Regularizado)
accuracy_mqo_alphas = {}

for alphaAtual in alphasValores:

    accuracies_mqo_alphas_rodadas = []  # Armazena acurácia para cada rodada

    for r in range(R):

        amostra_embaralhada = np.random.permutation(N)

        X = Data[amostra_embaralhada, :]
        Y = Y[amostra_embaralhada, :]

        # EMBARALHAR AS AMOSTRAS

        X_random = X[amostra_embaralhada,:]
        y_random = Y[amostra_embaralhada,:]

        # DIVIDIR TESTE E TREINO (EM X E Y) NA PROPORÇÃO 80-20

        X_treino = X_random[0:int(N*.8),:]  
        Y_treino = y_random[0:int(N*.8),:]  

        X_teste = X_random[int(N*.8):,:]
        Y_teste = y_random[int(N*.8):,:]

        XTX = X_treino.T @ X_treino
        ridge_model = np.linalg.inv(XTX + alphaAtual * np.identity(XTX.shape[0])) @ X_treino.T @ Y_treino

        y_pred_ridge = X_teste@ridge_model

        discriminante_teste = np.argmax(Y_teste, axis=1)

        # Calcular precisão para MQO Regularizado
        discriminante_ridge = np.argmax(y_pred_ridge, axis=1)
        accuracies_ridge = accuracy_score(discriminante_ridge, discriminante_teste)

        accuracies_mqo_alphas_rodadas.append(accuracies_ridge)
    
    # Armazena a media de acurácia
    accuracy_mqo_alphas[alphaAtual] = np.mean(accuracies_mqo_alphas_rodadas)

# Encontra o alpha com maior acurácia
# Para utilizar na implementação do MQO Regularizado
#best_alpha = max(accuracy_mqo_alphas, key=lambda alpha: accuracy_mqo_alphas[alpha])
best_alpha = max(accuracy_mqo_alphas)
#print(best_alpha)
print(best_alpha)

# Define a range of k values to test
k_values = np.arange(1, 9, 2)  # Adjust this list as needed

# Create a dictionary to store accuracies for each k
accuracies_knn_values = {}

for k in k_values:

    accuracy_knn_rodadas = []  # Armazena acurácia para cada rodada

    for r in range(R):
        amostra_embaralhada = np.random.permutation(N)
        X = Data[amostra_embaralhada, :]
        Y = Y[amostra_embaralhada, :]

        X = np.concatenate((
            np.ones((N, 1)), X
        ), axis=1)

        # EMBARALHAR AS AMOSTRAS
        X_random = X[amostra_embaralhada,:]
        y_random = Y[amostra_embaralhada,:]

        # DIVIDIR TESTE E TREINO (EM X E Y) NA PROPORÇÃO 80-20
        X_treino = X_random[0:int(N * .8),:]
        Y_treino = y_random[0:int(N * .8),:]
        X_teste = X_random[int(N * .8):,:]
        Y_teste = y_random[int(N * .8):,:]
    
        # k-NN model
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_treino, np.argmax(Y_treino, axis=1))
        Y_hat_knn = knn.predict(X_teste)

        # Calculating accuracy for k-NN
        discriminante_teste = np.argmax(Y_teste, axis=1)

        # Calcular precisão para MQO Regularizado
        accuracies_knn_teste = accuracy_score(Y_hat_knn, discriminante_teste)

        accuracy_knn_rodadas.append(accuracies_knn_teste)
    
    # Armazena a media de acurácia
    accuracies_knn_values[k] = np.mean(accuracy_knn_rodadas)

best_k = max(accuracies_knn_values)
print(best_k)

for r in range(R):
    amostra_embaralhada = np.random.permutation(N)

    X = Data[amostra_embaralhada, :]
    Y = Y[amostra_embaralhada, :]

    X = np.concatenate((
        np.ones((N, 1)), X
    ), axis=1)

    # EMBARALHAR AS AMOSTRAS
    
    X_random = X[amostra_embaralhada,:]
    y_random = Y[amostra_embaralhada,:]

    # DIVIDIR TESTE E TREINO (EM X E Y) NA PROPORÇÃO 80-20

    X_treino = X_random[0:int(N*.8),:]  
    Y_treino = y_random[0:int(N*.8),:]  

    X_teste = X_random[int(N*.8):,:]
    Y_teste = y_random[int(N*.8):,:]

    # ------------------------------

    # MQO Tradicional
    W_hat = np.linalg.pinv(X_treino.T @ X_treino) @ X_treino.T @ Y_treino
    Y_hat_lr = X_teste @ W_hat

    # Calcular precisão para MQO Tradicional
    discriminante_lr = np.argmax(Y_hat_lr, axis=1)
    discriminante2 = np.argmax(Y_teste, axis=1)
    accuracy_lr = accuracy_score(discriminante2, discriminante_lr)
    accuracies_lr.append(accuracy_lr)

    # ------------------------------

    # k vizinhos mais proximos
    k = best_k
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Use only features, excluding the bias term
    #knn.fit(X_treino, np.argmax(Y_treino, axis=1))

    Y_hat_knn = knn.predict(X_teste)

    # Calculando precisão para k vizinhos mais proximos
    accuracy_knn = accuracy_score(Y_hat_knn, discriminante2)
    accuracies_knn.append(accuracy_knn)

    # ------------------------------

    # DMC
    Y_hat_dmc = dmc_classifier(X_treino, np.argmax(Y_treino, axis=1), X_teste)

    # Calcular precisão para DMC
    accuracy_dmc = accuracy_score(Y_hat_dmc, discriminante2)
    accuracies_dmc.append(accuracy_dmc)

    # ------------------------------

    # MQO Regularizado

    XTX = X_treino.T @ X_treino
    W_hat_r = np.linalg.pinv(X_treino.T@X_treino + best_alpha*np.identity(XTX.shape[0]))@X_treino.T@Y_treino
    Y_hat_r = X_teste@W_hat_r

    discriminante_teste = np.argmax(Y_teste, axis=1)

    # Calcular precisão para MQO Regularizado
    discriminante_ridge = np.argmax(Y_hat_r, axis=1)
    accuracy_ridge = accuracy_score(discriminante_ridge, discriminante_teste)

    #accuracies_ridge.append(accuracy_ridge)

print(accuracies_dmc, accuracies_knn, accuracies_lr)


