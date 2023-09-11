import numpy as np
import matplotlib.pyplot as plt
plotarGraficos = True

#COLETA DE DADOS:
Data = np.loadtxt("sorvete.csv",delimiter=',')
#ORGANIZAÇÃO INICIAL (X e Y):

X = Data[:,0:2]
N,p = X.shape
y = Data[:,2].reshape(N,1)



#VISUALIZAÇÃO DOS DADOS:
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:,0],X[:,1],y[:,0],color='orange',edgecolors='k')
plt.show()



#PRÉ-PROCESSAMENTO?

#PROCESSAMENTO

#DEFINA A QUANTIDADE DE RODADAS
R = 1000

#Crie as listas vazias de desempenhos por rodada:

MSE_OLS_C =[]
MSE_OLS_S =[]
MSE_MEDIA =[]

for r in range(R):
    
    #Embaralhe as amostras de X e Y
    seed = np.random.permutation(N)
    X_random = X[seed,:]
    y_random = y[seed,:]
    #Divida X e Y em (X_treino,Y_treino) e (X_teste,Y_teste)  
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

    b_OLS_S = np.linalg.pinv(X_treino.T@X_treino)@X_treino.T@y_treino
    zero = np.zeros((1,1))
    b_OLS_S = np.concatenate((zero,b_OLS_S),axis=0)

    ones = np.ones((X_treino.shape[0],1))
    X_treino = np.concatenate((ones,X_treino),axis=1)
    b_OLS_C = np.linalg.pinv(X_treino.T@X_treino)@X_treino.T@y_treino


    if plotarGraficos:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(X_treino[:,1],X_treino[:,2],y_treino[:,0],color='purple',edgecolors='k')
        x_axis = np.linspace(0,40,500)
        y_axis = np.linspace(0,750,500)

        X_map,Y_map = np.meshgrid(x_axis,y_axis)
        X_map.shape = (500,500,1)
        Y_map.shape = (500,500,1)
        ones_map = np.ones((500,500,1))
        X3D = np.concatenate((ones_map,X_map,Y_map),axis=2)
        Z_media = X3D@b_media
        Z_ols_s = X3D@b_OLS_S
        Z_ols_c = X3D@b_OLS_C

        ax.plot_surface(X_map[:,:,0],Y_map[:,:,0],Z_media[:,:,0],cmap='gray',alpha=.5)
        ax.plot_surface(X_map[:,:,0],Y_map[:,:,0],Z_ols_s[:,:,0],cmap='winter',alpha=.5)
        ax.plot_surface(X_map[:,:,0],Y_map[:,:,0],Z_ols_c[:,:,0],cmap='jet',alpha=.5)


        
        plt.show()
    #Teste dos modelos, produzindo medida de erro/acerto 
    ones = np.ones((X_teste.shape[0],1))
    X_teste = np.concatenate((ones,X_teste),axis=1)
    
    y_pred_media = X_teste@b_media
    y_pred_ols_c = X_teste@b_OLS_C
    y_pred_ols_s = X_teste@b_OLS_S



    MSE_MEDIA.append(np.mean((y_teste-y_pred_media)**2))
    MSE_OLS_C.append(np.mean((y_teste-y_pred_ols_c)**2))
    MSE_OLS_S.append(np.mean((y_teste-y_pred_ols_s)**2))

    bp=1


boxplot = [MSE_MEDIA,MSE_OLS_C,MSE_OLS_S]
plt.boxplot(boxplot,labels=['Média','OLS com i','OLS sem i'])
plt.show()

#Compute a média, desvio padrão, maior valor e menor valor para cada modelo:

