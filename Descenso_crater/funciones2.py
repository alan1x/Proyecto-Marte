import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import random


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

    plt.scatter([x[0] for x in camino], [x[1] for x in camino], color='blue', s=9)
    plt.scatter(origen[0], origen[1], color='green', s=10,label='Origen')
    plt.scatter(camino[-1][0], camino[-1][1], color='black', s=10,label='Punto final')
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

def distancia(camino,scale):
    d=0
    for i in range(len(camino)-1):
        if (np.abs(camino[i][0]-camino[i+1][0]) + np.abs(camino[i][1]-camino[i+1][1]))==2:
            d+=2*scale
        else:
            d+=scale
    return d



def altura_nodo(matriz,nodo,scale=10.045):
    x,y=nodo
    ra,ca=cyr(matriz,x,y,scale)
    return matriz[ra,ca]

def diferencia_altura(matriz,nodo1,nodo2,altura=2):
    altura1=altura_nodo(matriz,nodo1)
    altura2=altura_nodo(matriz,nodo2)
    if altura1==-1 or altura2==-1:  
        return False 
    distancia=np.abs(altura1-altura2)
    if distancia<altura:
        return True
    else:
        return False



def obtener_vecinos(matriz, nodo):
    acciones = [(-1, 0), (1, 0), (0, -1), (0, 1),(1,1),(-1,1),(1,-1),(-1,-1)]  
    x,y=nodo
    while True:
        num = random.randint(0, 7)
        dx, dy = acciones[num]
        nx, ny = x + dx, y + dy
        if  diferencia_altura(matriz,nodo,(nx,ny),4):  
            return (nx, ny)
        else:
            continue


def es_mejor(matriz,nodo_origen,nodo_vecino):
    altura1=altura_nodo(matriz,nodo_origen)
    altura2=altura_nodo(matriz,nodo_vecino)
    if altura1 <= altura2:  
        return True 
    else:
        return False
    

def recocido(matriz, origen, t0, tf, alpha,k):
    actual = origen
    t = t0
    camino = [actual]
    while t > tf:
        vecino = obtener_vecinos(matriz, actual)
        if es_mejor(matriz, actual, vecino):
            actual = vecino
        else:
            p = random.random()
            if p < np.exp(k*(altura_nodo(matriz, actual) - altura_nodo(matriz, vecino)) / t):
                actual = vecino
        t = t * alpha
        camino.append(actual)


    return camino
