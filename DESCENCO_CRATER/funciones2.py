import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from collections import deque
import heapq
import random


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
    plt.scatter(origen[0], origen[1], color='green', s=10,label='Origen')
    plt.scatter(destino[0], destino[1], color='red', s=10,label='Destino')
    plt.xticks(np.arange(0, scale*nc, step=1000), rotation=45)
    plt.yticks(np.arange(0, scale*nr, step=1000))
    #plt.grid()
    plt.legend()
    plt.show()

def imagen_simple(matriz,nc,nr,scale):
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

    plt.xticks(np.arange(0, scale*nc, step=1000), rotation=45)
    plt.yticks(np.arange(0, scale*nr, step=1000))
    #plt.grid()
    plt.legend()
    plt.show()



def imagen2(camino,origen,matriz,nc,nr,scale=10.045):
    cmap = copy.copy(plt.cm.get_cmap('autumn'))
    cmap.set_under(color='black')   
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
    plt.scatter(origen[0], origen[1], color='green', s=10,label='Origen')
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

def rcy(matriz, r, c, scale):
    nr, nc = matriz.shape
    y = (nr - r) * scale  
    x = c * scale         
    return x, y



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
    if distancia<altura:
        return True
    else:
        return False
    
def obtener_vecinos(matriz, nodo):
    vecinos = []
    acciones = [(-1, 0), (1, 0), (0, -1), (0, 1),(1,1),(-1,1),(1,-1),(-1,-1)]  
    x, y = nodo
    for dx, dy in acciones:
        nx, ny = x + dx, y + dy
        if  diferencia_altura(matriz,nodo,(nx,ny)):  
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

def dfs(matriz, origen, objetivo, limite_profundidad=5000):
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




def altura_nodo(matriz,nodo,scale=10.045):
    x,y=nodo
    ra,ca=cyr(matriz,x,y,scale)
    return matriz[ra,ca]

def diferencia_altura2(matriz,nodo1,nodo2,altura=2):
    altura1=altura_nodo(matriz,nodo1)
    altura2=altura_nodo(matriz,nodo2)
    if altura1==-1 or altura2==-1:  
        return False 
    distancia=np.abs(altura1-altura2)
    if distancia<altura:
        return True
    else:
        return False



def obtener_vecinos2(matriz, nodo):
    acciones = [(-1, 0), (1, 0), (0, -1), (0, 1),(1,1),(-1,1),(1,-1),(-1,-1)]  
    x,y=nodo
    while True:
        num = random.randint(0, 7)
        dx, dy = acciones[num]
        nx, ny = x + dx, y + dy
        if  diferencia_altura2(matriz,nodo,(nx,ny)):  
            return (nx, ny)
        else:
            continue


def es_mejor(matriz,nodo_origen,nodo_vecino):
    altura1=altura_nodo(matriz,nodo_origen)
    altura2=altura_nodo(matriz,nodo_vecino)
    if altura1 >= altura2:  
        return False 
    else:
        return True
    

def recocido(matriz, origen, t0, tf, alpha):
    actual = origen
    t = t0
    camino = [actual]
    while t > tf:
        vecino = obtener_vecinos2(matriz, actual)
        if es_mejor(matriz, actual, vecino):
            actual = vecino
        else:
            p = random.random()
            if p < np.exp((altura_nodo(matriz, actual) - altura_nodo(matriz, vecino)) / t):
                actual = vecino
        t = t * alpha
        camino.append(actual)


    return camino