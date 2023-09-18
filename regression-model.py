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

# ORGANIZAR OS DADOS

N = X.shape[0]
X = X.reshape(N, 1)
y = y.reshape(N, 1)


# DEFININDO MODELOS
modelos = ['MQO Tradicional', 'Média Observável', 'MQO Regularizado']

# LISTAS DE DESEMPENHO POR RODADA (Erros quadraticos por modelo)

MSE_MQO_Tradicional = [] # Minimo de erros por rodada utilizando MQO Tradicional
MSE_MQO_Regularizado = [] # Minimo de erros por rodada utilizando MQO Regularizado
MSE_Media_Observavel = [] # Minimo de erros por rodada utilizando Media de valores oberservaveis
MSE_MQO_Regularizado_TestesAlpha = [] # Minimo de erros por rodada utilizando MQO Regularizado

# DEFININDO RODADAS

R = 1000
alphasValores = np.arange(0.001, 1.0, 0.001) # Logaritimo pequeno para amostra pequena, precisamos de um range pequeno para maior precisão do modelo

# Armazena erro quadratico para cada alpha (Apenas para o MQO Regularizado)
mse_mqo_alphas = {}
for alphaAtual in alphasValores:
    mse_mqo_alphas_rodadas = []  # Armazena o erro quadratico para cada rodada

    for r in range(R):

        #print(alphaAtual)

        # EMBARALHAR AS AMOSTRAS

        amostra_embaralhada = np.random.permutation(N)
        X_random = X[amostra_embaralhada,:]
        y_random = y[amostra_embaralhada,:]

        # DIVIDIR TESTE E TREINO (EM X E Y) NA PROPORÇÃO 80-20

        X_treino = X_random[0:int(N*.8),:]  
        y_treino = y_random[0:int(N*.8),:]  

        X_teste = X_random[int(N*.8):,:]
        y_teste = y_random[int(N*.8):,:]

        XTX = X_treino.T @ X_treino
        ridge_model = np.linalg.inv(XTX + alphaAtual * np.identity(XTX.shape[0])) @ X_treino.T @ y_treino
        
        y_pred_ridge = X_teste@ridge_model

        MSE_MQO_Regularizado_TestesAlpha.append(np.mean((y_teste - y_pred_ridge) ** 2))
    
    # Armazena a media de erro quadratico para esse alpha
    mse_mqo_alphas[alphaAtual] = np.mean(mse_mqo_alphas_rodadas)

# Encontra o alpha com menor erro quadratico
# Para utilizar na implementação do MQO Regularizado
best_alpha = min(mse_mqo_alphas)
min_mse = mse_mqo_alphas[best_alpha]

print(best_alpha)
print(min_mse)

for r in range(R):
        # EMBARALHAR AS AMOSTRAS

        amostra_embaralhada = np.random.permutation(N)
        X_random = X[amostra_embaralhada,:]
        y_random = y[amostra_embaralhada,:]

        #DIVIDIR TESTE E TREINO (EM X E Y) NA PROPORÇÃO 80-20

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

        MSE_Media_Observavel.append(np.mean((y_teste-y_pred_media)**2))
        MSE_MQO_Tradicional.append(np.mean((y_teste-y_pred_mqo_trad)**2))
        MSE_MQO_Regularizado.append(np.mean((y_teste - y_pred_ridge) ** 2))

boxplot = [MSE_MQO_Tradicional, MSE_Media_Observavel, MSE_MQO_Regularizado]
plt.boxplot(boxplot,labels=modelos)
plt.show()

# VALORES COMPUTADOS PARA A MEDIA, DESVIO PADRÃO, MENOR VALOR E MAIOR VALOR DOS MODELOS

# Métricas para o MQO Tradicional
media_MSE_MQO_Tradicional = np.mean(MSE_MQO_Tradicional)
desviopadrao_MSE_MQO_Tradicional = np.std(MSE_MQO_Tradicional)
menorvalor_MSE_MQO_Tradicional = min(MSE_MQO_Tradicional)
maiorvalor_MSE_MQO_Tradicional = max(MSE_MQO_Tradicional)

# Métricas para a Média Observável
media_MSE_Media_Observavel = np.mean(MSE_Media_Observavel)
desviopadrao_MSE_Media_Observavel = np.std(MSE_Media_Observavel)
menorvalor_MSE_Media_Observavel = min(MSE_Media_Observavel)
maiorvalor_MSE_Media_Observavel = max(MSE_Media_Observavel)

# Métricas para o MQO Regularizado
media_MSE_MQO_Regularizado = np.mean(MSE_MQO_Regularizado)
desviopadrao_MSE_MQO_Regularizado = np.std(MSE_MQO_Regularizado)
menorvalor_MSE_MQO_Regularizado = min(MSE_MQO_Regularizado)
maiorvalor_MSE_MQO_Regularizado = max(MSE_MQO_Regularizado)

# Valores para cada métrica
medias = [media_MSE_MQO_Tradicional, media_MSE_Media_Observavel, media_MSE_MQO_Regularizado]
desvios_padrao = [desviopadrao_MSE_MQO_Tradicional, desviopadrao_MSE_Media_Observavel, desviopadrao_MSE_MQO_Regularizado]
menor_valores = [menorvalor_MSE_MQO_Tradicional, menorvalor_MSE_Media_Observavel, menorvalor_MSE_MQO_Regularizado]
maior_valores = [maiorvalor_MSE_MQO_Tradicional, maiorvalor_MSE_Media_Observavel, maiorvalor_MSE_MQO_Regularizado]

largura_barras = 0.2
# Posições das barras no eixo x
posicoes_m1 = np.arange(len(modelos))
posicoes_m2 = [x + largura_barras for x in posicoes_m1]
posicoes_m3 = [x + largura_barras for x in posicoes_m2]
posicoes_m4 = [x + largura_barras for x in posicoes_m3]

# Criar o gráfico de barras
plt.bar(posicoes_m1, medias, largura_barras, label='Média')
plt.bar(posicoes_m2, desvios_padrao, largura_barras, label='Desvio Padrão')
plt.bar(posicoes_m3, menor_valores, largura_barras, label='Menor Valor')
plt.bar(posicoes_m4, maior_valores, largura_barras, label='Maior Valor')

# Adicionar detalhes ao gráfico
plt.xlabel('Modelos')
plt.ylabel('Valores')
plt.title('Comparação das Métricas para os Modelos')
plt.xticks(posicoes_m2, modelos)
plt.legend()

# Mostrar o gráfico
plt.tight_layout()
plt.show()


