import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from collections import deque
import heapq



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

    plt.scatter([x[0] for x in camino], [x[1] for x in camino], color='blue', s=10)
    plt.scatter(origen[0], origen[1], color='green', s=10)
    plt.scatter(destino[0], destino[1], color='red', s=10)
    plt.xticks(np.arange(0, scale*nc, step=1000), rotation=45)
    plt.yticks(np.arange(0, scale*nr, step=1000))
    #plt.grid()
    plt.show()

def cyr(matriz,x,y,scale):
    nr,nc=matriz.shape
    r=nr-round(y/scale)
    c=round(x/scale)
    return r,c

def distancia(camino,scale):
    d=0
    for i in range(len(camino)-1):
        if (np.abs(camino[i][0]-camino[i+1][0]) + np.abs(camino[i][1]-camino[i+1][1]))==2:
            d+=2*scale
        else:
            d+=scale
    return d

def diferencia_altura(matriz,nodo1,nodo2,scale=10.0174,altura=1.25):
    r1,c1=nodo1
    ra1,ca1=cyr(matriz,c1,r1,scale)
    r2,c2=nodo2
    ra2,ca2=cyr(matriz,c2,r2,scale)
    altura1=matriz[ra1,ca1]
    altura2=matriz[ra2,ca2]
    if altura1==-1 or altura2==-1:
        return False 
    distancia=np.abs(altura1-altura2)
    if distancia<=altura:
        return True
    else:
        return False
    
def obtener_vecinos(matriz, nodo):
    vecinos = []
    acciones = [(-1, 0), (1, 0), (0, -1), (0, 1),(1,1),(-1,1),(1,-1),(-1,-1)]  
    x, y = nodo
    for dx, dy in acciones:
        nx, ny = x + dx, y + dy
        if  diferencia_altura(matriz,nodo,(nx,ny),10.0174,1.25):  
            vecinos.append((nx, ny))
    return vecinos

def bfs(matriz, origen, objetivo):
    explorados = set()
    frontera = deque([origen])
    padres = {origen: None}  

    while frontera:
        inicio = frontera.popleft()
        if inicio == objetivo:
            break
        explorados.add(inicio)
        for vecino in obtener_vecinos(matriz, inicio):
            if vecino not in explorados and vecino not in frontera:
                frontera.append(vecino)
                padres[vecino] = inicio  
    camino = []
    paso = objetivo
    while paso is not None:
        camino.append(paso)
        paso = padres[paso]
    camino.reverse() 

    return camino

def h(nodo1,objetivo):
    x1,y1=nodo1
    x2,y2=objetivo
    return np.abs(x1-x2)+np.abs(y1-y2)

def g(origen,nodo1):
    x1,y1=origen
    x2,y2=nodo1
    return np.abs(x1-x2)+np.abs(y1-y2)

def a_estrella(matriz, origen, objetivo):
    frontera = []
    heapq.heappush(frontera, (0, origen))
    padres = {origen: None}
    g_costos = {origen: 0}

    while frontera:
        actual = heapq.heappop(frontera)[1]

        if actual == objetivo:
            break

        for vecino in obtener_vecinos(matriz, actual):
            nuevo_costo = g_costos[actual] + 1  
            if vecino not in g_costos or nuevo_costo < g_costos[vecino]:
                g_costos[vecino] = nuevo_costo
                prioridad = nuevo_costo + h(vecino, objetivo)
                heapq.heappush(frontera, (prioridad, vecino))
                padres[vecino] = actual

    if objetivo not in padres:
        return None  

    camino = []
    paso = objetivo
    while paso is not None:
        camino.append(paso)
        paso = padres[paso]
    camino.reverse()

    return camino    