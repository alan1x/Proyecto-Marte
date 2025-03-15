import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource


def imagen(camino,origen,destino,matriz,nc,nr):
    cmap = copy.copy(plt.cm.get_cmap('autumn'))
    cmap.set_under(color='black')   
    scale=10.0174
    ls = LightSource(315, 45)
    rgb = ls.shade(matriz, cmap=cmap, vmin = 0, vmax = matriz.max(), vert_exag=2, blend_mode='hsv')

    fig, ax = plt.subplots()

    im = ax.imshow(rgb, cmap=cmap, vmin = 0, vmax = matriz.max(), 
                    extent =[0, scale*nc, 0, scale*nr], 
                    interpolation ='nearest', origin ='upper')

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Altura (m)')

    plt.title('Superficie de Marte')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')

    plt.scatter([x[0] for x in camino], [x[1] for x in camino], color='blue', s=1)
    plt.scatter(origen[0], origen[1], color='green', s=10,label='Origen')
    plt.scatter(destino[0], destino[1], color='red', s=10,label='Destino')
    plt.xticks(np.arange(0, scale*nc, step=1000), rotation=45)
    plt.yticks(np.arange(0, scale*nr, step=1000))
    #plt.grid()
    plt.legend()
    plt.show()


def cyr(matriz,x,y,scale):
    nr,nc=matriz.shape
    r=nr-round(y/scale)
    c=round(x/scale)
    return r,c



def diferencia_altura(matriz, nodo1, nodo2):
    r1, c1 = nodo1
    r2, c2 = nodo2
    # Esto lo ponemos para checar que los índices no estén fuera de la matriz
    if not (0 <= r1 < matriz.shape[0] and 0 <= c1 < matriz.shape[1]):
        return False
    if not (0 <= r2 < matriz.shape[0] and 0 <= c2 < matriz.shape[1]):
        return False
    altura1 = matriz[r1, c1]
    altura2 = matriz[r2, c2]
     # checa  si alguna altura es -1  que no se puede estar en ese terreno y devuelve false
    if altura1 == -1 or altura2 == -1:
        return False
    distancia = np.abs(altura1 - altura2)
    return distancia < 0.25



def obtener_vecinos(matriz, nodo):
    vecinos = []
    acciones = [(-1, 0), (1, 0), (0, -1), (0, 1),(1,1),(-1,1),(1,-1),(-1,-1)]  
    r, c = matriz.shape
    x, y = nodo
    for dx, dy in acciones:
        nx, ny = x + dx, y + dy
        if  diferencia_altura(matriz,nodo,(nx,ny)):  
            vecinos.append((nx, ny))
    return vecinos


def rcy(matriz, r, c, scale):
    nr, nc = matriz.shape
    y = (nr - r) * scale  
    x = c * scale         
    return x, y

def dfs(matriz, origen, objetivo, limite_profundidad=5000, scale=10.0174):
    stack = [(origen, 0)] 
    explorados = set()  
    padres = {origen: None}  

    while stack:
        inicio, profundidad = stack.pop()

        if profundidad > limite_profundidad:
            continue

        if inicio == objetivo:
            break

        if inicio not in explorados:
            explorados.add(inicio)

            for vecino in obtener_vecinos(matriz, inicio):
                if vecino not in explorados and vecino not in [x[0] for x in stack]:
                    stack.append((vecino, profundidad + 1))
                    padres[vecino] = inicio

    if objetivo not in padres:
        return []

    camino = []
    paso = objetivo
    while paso is not None:
        camino.append(paso)
        paso = padres.get(paso, None)

    camino.reverse()
    camino_real = [rcy(matriz, r, c, scale) for r, c in camino]
    return camino_real

