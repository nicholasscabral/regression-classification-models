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
plt.show()

# ORGANIZAR OS DADOS

N = X.shape[0]
print(N)
X = X.reshape(N, 1)
y = y.reshape(N, 1)


# DEFINITINDO RODADAS

R = 1000

for r in range(R):
    