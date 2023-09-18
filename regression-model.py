import numpy as np
import matplotlib.pyplot as plt

#COLETA DE DADOS:
Data = np.loadtxt("sorvete.csv",delimiter=',', skiprows=1)

# DEFINITINDO VARIAVEL REGRESSORA E OBERSEVARDORA

X = Data[0:,0]
y = Data[0:,1]

# VISUALIZAÇÃO DOS DADOS:

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(X, y, color='red',edgecolors='k')
#plt.show()

# ORGANIZAR OS DADOS

N = X.shape[0]
X = X.reshape(N, 1)
y = y.reshape(N, 1)


# LISTAS DE DESEMPENHO POR RODADA

MSE_MQOTradicional =[]

MSE_Media =[]

MSE_Ridge = []

# DEFININDO RODADAS

R = 1000


# Range of alpha values to search
alphas = np.logspace(0.001, 1.0)  # You can adjust the range as needed

# Dictionary to store the MSE for each alpha
mse_dict = {}

for alpha in alphas:
    mse_list = []  # Store MSE values for each round

    for r in range(R):
        
        # EMBARALHAR AS AMOSTRAS

        amostra_embaralhada = np.random.permutation(N)
        X_random = X[amostra_embaralhada,:]
        y_random = y[amostra_embaralhada,:]

        #DIVIDIR TESTE E TREINO (EM X E Y) NA PROPORÇÃO 20-80

        X_treino = X_random[0:int(N*.8),:]  
        y_treino = y_random[0:int(N*.8),:]  

        X_teste = X_random[int(N*.8):,:]
        y_teste = y_random[int(N*.8):,:]

        XTX = X_treino.T @ X_treino
        ridge_model = np.linalg.inv(XTX + alpha * np.identity(XTX.shape[0])) @ X_treino.T @ y_treino
        
        y_pred_ridge = X_teste@ridge_model

        MSE_Ridge.append(np.mean((y_teste - y_pred_ridge) ** 2))
    
    # Store the average MSE for this alpha
    mse_dict[alpha] = np.mean(mse_list)

# Find the alpha with the minimum MSE
best_alpha = min(mse_dict)
min_mse = mse_dict[best_alpha]

print(best_alpha)
print(min_mse)


for r in range(R):
        
        # EMBARALHAR AS AMOSTRAS

        amostra_embaralhada = np.random.permutation(N)
        X_random = X[amostra_embaralhada,:]
        y_random = y[amostra_embaralhada,:]

        #DIVIDIR TESTE E TREINO (EM X E Y) NA PROPORÇÃO 20-80

        X_treino = X_random[0:int(N*.8),:]  
        y_treino = y_random[0:int(N*.8),:]  

        X_teste = X_random[int(N*.8):,:]
        y_teste = y_random[int(N*.8):,:]


        #Treinamento dos modelos  
        modelo_media = np.mean(y_treino)
        modelo_media = np.array([
            [modelo_media],
            [0],
        ])
        
        ones = np.ones((X_treino.shape[0],1))
        X_treino = np.concatenate((ones,X_treino),axis=1)
        modelo_MQO_trad = np.linalg.pinv(X_treino.T@X_treino)@X_treino.T@y_treino 

        XTX = X_treino.T @ X_treino
        ridge_model = np.linalg.inv(XTX + best_alpha * np.identity(XTX.shape[0])) @ X_treino.T @ y_treino

        #Teste dos modelos, produzindo medida de erro/acerto 
        ones = np.ones((X_teste.shape[0],1))
        X_teste = np.concatenate((ones,X_teste),axis=1)
        
        y_pred_media = X_teste@modelo_media
        y_pred_mqo_trad = X_teste@modelo_MQO_trad
        y_pred_ridge = X_teste@ridge_model

        MSE_Media.append(np.mean((y_teste-y_pred_media)**2))
        MSE_MQOTradicional.append(np.mean((y_teste-y_pred_mqo_trad)**2))
        MSE_Ridge.append(np.mean((y_teste - y_pred_ridge) ** 2))

boxplot = [MSE_Media,MSE_MQOTradicional, MSE_Ridge]
plt.boxplot(boxplot,labels=['Média','MQO', 'Ridge'])
plt.show()

#VALORES COMPUTADOS PARA A MEDIA, DESVIO PADRÃO, MENOR VALOR E MAIOR VALOR DOS MODELOS
media_MQOTradicional = np.mean(MSE_MQOTradicional)
media_MediaObservavel = np.mean(MSE_Media)
media_MQORegularizado = np.mean(MSE_Ridge)
desviopadrao_MQOTradicional = np.std(MSE_MQOTradicional)
desviopadrao_Media = np.std(MSE_Media)
desviopadrao_MQORegularizado = np.std(MSE_Ridge)
menorvalor_MQOTradicional = min(MSE_MQOTradicional)
menorvalor_Media = min(MSE_Media)
menorvalor_MQORegularizado = min(MSE_Ridge)
maiorvalor_MQOTradicional = max(MSE_MQOTradicional)
maiorvalor_Media = max(MSE_MQOTradicional)
maiorvalor_MQOTradicional = max(MSE_MQOTradicional)


