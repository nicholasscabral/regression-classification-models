import numpy as np
import matplotlib.pyplot as plt
plotarGraficos = False

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
print(N)
X = X.reshape(N, 1)
y = y.reshape(N, 1)


# LISTAS DE DESEMPENHO POR RODADA

#Modelo com interceptor (MQO tradicional)
MSE_OLS_C =[]
#Modelo sem interceptor (MQO regularizado)
MSE_OLS_S =[]
#MODELO: Média de valores observáveis
MSE_MEDIA =[]

# DEFININDO RODADAS

#R=1000
R = 1

for r in range(R):
    
    # EMBARALHAR AS AMOSTRAS

    seed = np.random.permutation(N)
    X_random = X[seed,:]
    y_random = y[seed,:]

    #DIVIDIR TESTE E TREINO (EM X E Y) NA PROPORÇÃO 20-80

    X_treino = X_random[0:int(N*.8),:]  
    y_treino = y_random[0:int(N*.8),:]  

    X_teste = X_random[int(N*.8):,:]
    y_teste = y_random[int(N*.8):,:]
    
    
    bp=1

    #Treinamento dos modelos  
    b_media = np.mean(y_treino)
    b_media = np.array([
        [b_media],
        [0],
        [0]
    ])


    b_OLS_S = np.linalg.pinv(X_treino.T@X_treino)@X_treino.T@y_treino #modelo
    zero = np.zeros((1,1))
    b_OLS_S = np.concatenate((zero,b_OLS_S),axis=0)
    
    ones = np.ones((X_treino.shape[0],1))
    X_treino = np.concatenate((ones,X_treino),axis=1)
    b_OLS_C = np.linalg.pinv(X_treino.T@X_treino)@X_treino.T@y_treino #modelo com interceptor

    #posso colocar um contador pra ir até 1000, plotando todos os gráficos

    #if plotarGraficos:
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(X_treino[:,1],y_treino[:,0],color='purple',edgecolors='k')
    plt.show()
    
    
    
    x_axis = np.linspace(0,40,500)
    y_axis = np.linspace(0,750,500)
    plt.show()
    #Consigo printar o treinamento
    

    '''
    PARTE QUE NÃO ESTÁ FUNCIONANDO!
    A DIMENSÃO DAS MATRIZES QUE SE MULTIPLICAM TÁ ERRADA

    X_map,Y_map = np.meshgrid(x_axis,y_axis)
    X_map.shape = (500,500,1)
    Y_map.shape = (500,500,1)
    ones_map = np.ones((500,500,1))
    X3D = np.concatenate((ones_map,X_map,Y_map),axis=1)
    Z_media = X3D@b_media
    Z_ols_s = X3D@b_OLS_S
    Z_ols_c = X3D@b_OLS_C

    ax.plot_surface(X_map[:,:,0],Y_map[:,:,0],Z_media[:,:,0],cmap='gray',alpha=.5)
    ax.plot_surface(X_map[:,:,0],Y_map[:,:,0],Z_ols_s[:,:,0],cmap='winter',alpha=.5)
    ax.plot_surface(X_map[:,:,0],Y_map[:,:,0],Z_ols_c[:,:,0],cmap='jet',alpha=.5)
    
    plt.show()'''
