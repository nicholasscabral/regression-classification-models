import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

# FUNÇÃO CLASSIFICADORA DO CENTROIDE (para cada label)
def dmc_classifier(X_treino, Y_treino, X_teste):
    centroides = []
    for label in np.unique(Y_treino):
        amostras_rotulo = X_treino[Y_treino == label]
        centroide = np.mean(amostras_rotulo, axis=0)
        centroides.append(centroide)
    centroides = np.array(centroides)

    Y_pred = np.argmin(np.linalg.norm(X_teste[:, np.newaxis] - centroides, axis=2), axis=1)
    return Y_pred

'''
def best_k(Data, R, X, Y):

    # Define a range of k values to test
    k_values = np.arange(1, 10, 2)  # Adjust this list as needed

    # Create a dictionary to store accuracies for each k
    accuracies_knn_values = {}

    for k in k_values:

        accuracy_knn_rodadas = []  # Armazena acurácia para cada rodada

        for r in range(R):

            s = np.random.permutation(N)

            X = Data[s,:]
            Y = Y[s,:]

            X = np.concatenate((
                np.ones((N,1)),X
            ),axis=1)

            X_treino = X[0:int(N*.8),:]
            Y_treino = Y[0:int(N*.8),:]

            X_teste = X[int(N*.8):,:]
            Y_teste = Y[int(N*.8):,:]
        
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

    best_k = max(accuracies_knn_values, key=lambda k: accuracies_knn_values[k])
    return best_k

def best_alpha(Data, R, X, Y):

    alphasValores = np.arange(0.1, 1.1, 0.1) # Logaritimo pequeno para amostra pequena, precisamos de um range pequeno para maior precisão do modelo

    # Armazena erro quadratico para cada alpha (Apenas para o MQO Regularizado)
    mse_mqo_alphas = {}

    for alphaAtual in alphasValores:

        mse_mqo_alphas_rodadas = []  # Armazena o erro quadratico para cada rodada

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

            ridge = Ridge(alpha=alphaAtual)
            ridge.fit(X_treino[:, 1:], Y_treino)

            Y_hat_ridge = ridge.predict(X_teste[:, 1:])

            mse_mqo_alphas_rodadas.append(np.mean((Y_teste - Y_hat_ridge) ** 2))

        # Armazena a media de erro quadratico para esse alpha
        mse_mqo_alphas[alphaAtual] = np.mean(mse_mqo_alphas_rodadas)

    # Encontra o alpha com menor erro quadratico
    # Para utilizar na implementação do MQO Regularizado
    best_alpha = min(mse_mqo_alphas, key=lambda k: mse_mqo_alphas[k])
    min_mse = mse_mqo_alphas[best_alpha]

    return best_alpha
'''


# RECEBER DADOS DE ENTRADA
Data = np.loadtxt("EMG.csv", delimiter=',')
N, p = Data.shape
colors = ['red', 'green', 'purple', 'blue', 'gray']


# DEFINIR ROTULOS DE CLASSIFICAÇÃO
neutro = np.tile(np.array([[1,-1,-1,-1,-1]]),(1000,1)) 
sorrindo = np.tile(np.array([[-1,1,-1,-1,-1]]),(1000,1)) 
aberto = np.tile(np.array([[-1,-1,1,-1,-1]]),(1000,1)) 
surpreso = np.tile(np.array([[-1,-1,-1,1,-1]]),(1000,1)) 
rabugento = np.tile(np.array([[-1,-1,-1,-1,1]]),(1000,1)) 
Y = np.tile(np.concatenate((neutro,sorrindo,aberto,surpreso,rabugento)),(10,1))

# NUMERO DE RODADAS
R = 100

precisao_mqo_tradicional = []
precisao_knn = []
precisao_dmc = []
precisao_mqo_regularizado = []


for r in range(R):

    s = np.random.permutation(N)

    X = Data[s,:]
    Y = Y[s,:]

    X = np.concatenate((
        np.ones((N,1)),X
    ),axis=1)

    X_treino = X[0:int(N*.8),:]
    Y_treino = Y[0:int(N*.8),:]

    X_teste = X[int(N*.8):,:]
    Y_teste = Y[int(N*.8):,:]

    # MODELO MQO TRADICIONAL

    # IMPLEMENTAÇÃO
    W_hat = np.linalg.pinv(X_treino.T@X_treino)@X_treino.T@Y_treino
    Y_hat = X_teste@W_hat

    # PRECISÃO
    discriminante = np.argmax(Y_hat,axis=1)
    discriminante2 = np.argmax(Y_teste,axis=1)

    acertos = discriminante==discriminante2
    contagem_acertos = np.count_nonzero(acertos)/10000
    precisao_mqo_tradicional.append(contagem_acertos)

    # ------------------------------

    # MQO Regularizado

    XTX = X_treino.T @ X_treino
    W_hat_r = np.linalg.pinv(X_treino.T@X_treino + 0.1*np.identity(XTX.shape[0]))@X_treino.T@Y_treino
    Y_hat_r = X_teste@W_hat_r

    # Calcular precisão para MQO Regularizado
    discriminante_ridge = np.argmax(Y_hat_r, axis=1)
    acertos_ridge = discriminante_ridge==discriminante2
    contagem_acertos_ridge = np.count_nonzero(acertos_ridge)/10000
    precisao_mqo_regularizado.append(contagem_acertos_ridge)

    # ------------------------------

    # MODELO K-NN (K VIZINHOS MAIS PRÓXIMOS)

    # IMPLEMENTAÇÃO
    k = 3
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_treino, Y_treino)
    
    # PRECISÃO
    Y_hat_knn = knn.predict(X_teste)
    accuracy = accuracy_score(Y_teste, Y_hat_knn)
    precisao_knn.append(accuracy)

    # ------------------------------

    # MODELO DMC (DISTÂNCIA AO CENTROIDE)

    # IMPLEMENTAÇÃO
    dmc = dmc_classifier(X_treino, np.argmax(Y_treino, axis=1), X_teste)

    # PRECISÃO
    accuracy_dmc = accuracy_score(dmc, discriminante2)
    precisao_dmc.append(accuracy_dmc)

    # ------------------------------

print(precisao_mqo_tradicional, precisao_mqo_regularizado, precisao_knn, precisao_dmc)


# Calculate mean and standard deviation of accuracy for each model
mean_accuracy_mqo_tradicional = np.mean(precisao_mqo_tradicional)
std_accuracy_mqo_tradicional = np.std(precisao_mqo_tradicional)

# Calculate mean and standard deviation for other models similarly

print("Mean Accuracy (MQO Tradicional):", mean_accuracy_mqo_tradicional)
print("Standard Deviation (MQO Tradicional):", std_accuracy_mqo_tradicional)
