import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Vamos supor que o arquivo está no mesmo diretório do script e se chama 'dados.csv'
Data = pd.read_csv('EMG.csv', header=None,names=['Sensor1', 'Sensor2'])

#Definição para as variaveis X e Y
X = Data.values
N,p = X.shape


neutro = np.tile(np.array([[1,-1,-1,-1,-1]]),(1000,1)) 
sorrindo = np.tile(np.array([[-1,1,-1,-1,-1]]),(1000,1)) 
aberto = np.tile(np.array([[-1,-1,1,-1,-1]]),(1000,1)) 
surpreso = np.tile(np.array([[-1,-1,-1,1,-1]]),(1000,1)) 
rabugento = np.tile(np.array([[-1,-1,-1,-1,1]]),(1000,1)) 
Y = np.tile(np.concatenate((neutro,sorrindo,aberto,surpreso,rabugento)),(10,1))


def bestAlpha(rounds, X, Y):
    # Gere N valores no intervalo 0 < λ ≤ 1
    alphaValues =  np.arange(0.1, 1.01, 0.1) 
    bestAlpha = 1
    maxValue = -1

    for currentAlpha in alphaValues:

        accuracies_alpha = []

        #print("Alfa atual: ", alphaAtual)
        for round in range(rounds):
            indexRandom = np.random.permutation(N)
            indexOfOitentaPorCento = int(N*.8)

            #Embaralhar dados
            X_embaralhado = X[indexRandom,:]
            Y_embaralhado = Y[indexRandom,:]

            #6. Amostra para treino e teste 
            X_treino = X_embaralhado[0: indexOfOitentaPorCento,:] #Ir de Zero até o index 80% total (no caso é 39)
            Y_treino = Y_embaralhado[0: indexOfOitentaPorCento,:]
            X_teste =  X_embaralhado[indexOfOitentaPorCento: N,:] #Ir do ultimo index que representa os 80% até o fim
            Y_teste =  Y_embaralhado[indexOfOitentaPorCento: N,:]

            modelo_mqo_regularizado = np.linalg.inv((X_treino.T @ X_treino) + currentAlpha * np.identity((X_treino.T @ X_treino).shape[0]))@ X_treino.T @ Y_treino

            Y_predicao = X_teste @ modelo_mqo_regularizado

            descriminante_predicao = np.argmax(Y_predicao, axis=1)
            descriminante_teste = np.argmax(Y_teste, axis=1)
            acuaria_mqo_regularizado = accuracy_score(descriminante_predicao, descriminante_teste)

            accuracies_alpha.append(acuaria_mqo_regularizado)
        
        if(np.mean(accuracies_alpha) > maxValue):
            maxValue = np.mean(accuracies_alpha)
            bestAlpha = currentAlpha

    return bestAlpha

def determinarAcuracia(X_Teste, Y_teste,MODELO, label):
    Y_predicao = X_Teste @ MODELO

    descriminante_predicao = np.argmax(Y_predicao, axis=1)
    descriminante_teste = np.argmax(Y_teste, axis=1)
    acuracia_modelo = accuracy_score(descriminante_predicao, descriminante_teste)

    if(label != ""):
        print("Modelo: " , label ,", Acurácia: " , acuracia_modelo , "\n")

    return acuracia_modelo
    
def distancia_euclidiana(x1, x2):
    """Calcula a distância euclidiana entre dois pontos."""
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_classificador(X_treino, y_treino, X_teste, k):
    """Classificador k-NN simples."""
    y_pred = []
    for i in range(len(X_teste)):
        print(i)
        distancias = [distancia_euclidiana(X_treino[j], X_teste[i]) for j in range(len(X_treino))]
        indices_vizinhos = np.argsort(distancias)[:k]
        vizinhos = [y_treino[idx] for idx in indices_vizinhos]
        
        # Encontre a classe mais frequente usando a função numpy unique
        classes, counts = np.unique(vizinhos, return_counts=True)
        classe_mais_frequente = classes[np.argmax(counts)]
        
        y_pred.append(classe_mais_frequente)
    return np.array(y_pred)

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

# 2
RODADAS_DE_TREINAMENTO = 3

##Modelos para implementação
MQO_TRADICIONAL = [] #Modelo com intercepitor
MQO_REGULARIZADO = []

melhorAlpha = bestAlpha(RODADAS_DE_TREINAMENTO, X , Y)

acuracia_MQO_TRADICIONAL_registros = []
acuracia_MQO_REGULARIZADO_registros = []
acuracia_KNN_registros = []
acuracia_DMC_registros = []

interceptorTreino = np.ones((X.shape[0] , 1)) 
X = np.concatenate((interceptorTreino , X),axis=1)

for rodada in range(RODADAS_DE_TREINAMENTO):
    indexRandom = np.random.permutation(N)
    indexOfOitentaPorCento = int(N*.8)

    #Embaralhar dados
    X_embaralhado = X[indexRandom,:]
    Y_embaralhado = Y[indexRandom,:]

    #6. Amostra para treino e teste 
    X_treino = X_embaralhado[0: indexOfOitentaPorCento,:] #Ir de Zero até o index 80% total (no caso é 39)
    Y_treino = Y_embaralhado[0: indexOfOitentaPorCento,:]
    X_teste =  X_embaralhado[indexOfOitentaPorCento: N,:] #Ir do ultimo index que representa os 80% até o fim
    Y_teste =  Y_embaralhado[indexOfOitentaPorCento: N,:]

        #Modelo MQO regularizado 
    MODELO_MQO_REGULARIZADO = np.linalg.inv((X_treino.T @ X_treino) + melhorAlpha * np.identity((X_treino.T @ X_treino).shape[0]))@ X_treino.T @ Y_treino
    acuracia_mqo_regularizado = determinarAcuracia(X_teste, Y_teste, MODELO_MQO_REGULARIZADO, "")
    acuracia_MQO_REGULARIZADO_registros.append(acuracia_mqo_regularizado)
    print(acuracia_MQO_REGULARIZADO_registros)


    #Modelo MQO tradicional - com interceptor
    MODELO_MQO_TRADICIONAL = np.linalg.pinv(X_treino.T@X_treino)@X_treino.T@Y_treino
    acuracia_mqo_tradicional = determinarAcuracia(X_teste, Y_teste, MODELO_MQO_TRADICIONAL, "")
    acuracia_MQO_TRADICIONAL_registros.append(acuracia_mqo_tradicional)
    print(acuracia_MQO_TRADICIONAL_registros)

    # MODELO K-NN (K VIZINHOS MAIS PRÓXIMOS)

    # IMPLEMENTAÇÃO
    k = 3
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_treino, Y_treino)
    
    # PRECISÃO
    Y_hat_knn = knn.predict(X_teste)
    accuracy = accuracy_score(Y_teste, Y_hat_knn)
    acuracia_KNN_registros.append(accuracy)
    print(acuracia_KNN_registros)

    # MODELO DMC (DISTÂNCIA AO CENTROIDE)

    # IMPLEMENTAÇÃO
    dmc = dmc_classifier(X_treino, np.argmax(Y_treino, axis=1), X_teste)
    discriminante_teste = np.argmax(Y_teste,axis=1)

    # PRECISÃO
    accuracy_dmc = accuracy_score(dmc, discriminante_teste)
    acuracia_DMC_registros.append(accuracy_dmc)
    print(acuracia_DMC_registros)
    