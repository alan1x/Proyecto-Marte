import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from collections import deque
import heapq
import heapq

class GreedyBestFirstSearch:
    def __init__(self, problem):
        self.problem = problem
        self.frontier = []
        heapq.heappush(self.frontier, (self.heuristic(problem.initial), problem.initial))
        self.explored = set()
        self.came_from = {}
    
    # Heurística: distancia Manhattan que cuenta las diagonales
    def heuristic(self, state):
        x1, y1 = state
        x2, y2 = self.problem.goal
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        return (dx + dy) + (np.sqrt(2) - 2) * min(dx, dy)

    def search(self):
        while self.frontier:
            _, current_state = heapq.heappop(self.frontier)
            if self.problem.goal_test(current_state):
                return self.solution(current_state)
            self.explored.add(current_state)
            
            for action in self.problem.actions(current_state):
                child_state = self.problem.result(current_state, action)
                if child_state not in self.explored:
                    heapq.heappush(self.frontier, (self.heuristic(child_state), child_state))
                    self.came_from[child_state] = current_state
        return None

    def solution(self, state):
        path = []
        while state in self.came_from:
            path.append(state)
            state = self.came_from[state]
        path.append(state)
        path.reverse()
        return path


class Marte_Problem_Crater:
    def __init__(self, initial, maze):
        self.initial = initial
        self.maze = maze
        self.rows, self.cols = maze.shape
        self.goal = self.find_lowest_point()

    def find_lowest_point(self):
        valid_values = self.maze[self.maze > 1]
        if valid_values.size == 0:
            raise ValueError("No se encontró ningún punto válido en el mapa.")
        min_value = np.min(valid_values)
        min_indices = np.where(self.maze == min_value)
        return list(zip(min_indices[0], min_indices[1]))[0]

    def goal_test(self, state):
        return state == self.goal

    def actions(self, state):
        x, y = state
        posibles_mov = [
            (x-1, y), 
            (x+1, y),   
            (x, y-1),   
            (x, y+1),   
            (x-1, y-1), 
            (x-1, y+1), 
            (x+1, y-1), 
            (x+1, y+1)  
        ]

        acciones_validas = []
        altura_actual = self.maze[x, y]
        for nx, ny in posibles_mov:
            if 0 <= nx < self.rows and 0 <= ny < self.cols and self.maze[nx, ny] != 255:
                altura_nueva = self.maze[nx, ny]
                if abs(altura_nueva - altura_actual) < 2:
                    acciones_validas.append((nx, ny))
        return acciones_validas

    def result(self, state, action):
        return action

def calxy(nr, nc, r, c, scale):
    y = (nr - r) * scale
    x = c * scale
    return x, y

def camino_recorrido_crater(crater_map, x_start ,y_start):
    

    def calcrc(nr, nc, x, y, scale): 
        r = nr - round(y / scale)
        c = round(x / scale)
        return r, c

    scale = 10.0174
    nr, nc = crater_map.shape  

    start = calcrc(nr, nc, x_start, y_start, scale)

    problem = Marte_Problem_Crater(start, crater_map)
    gbfs = GreedyBestFirstSearch(problem)
    solution = gbfs.search()
    solution_scaled = [calxy(nr, nc, r, c, scale) for r, c in solution]
    print(f"La solución es: {solution_scaled}")
    print(f"Longitud del camino: {len(solution_scaled)}")
    print(f"Altura de la meta: {crater_map[problem.goal[0], problem.goal[1]]}")
    print(f"Altura del camino recorrido: {crater_map[start[0], start[1]]}")

    cmap = copy.copy(plt.cm.get_cmap('autumn'))
    cmap.set_under(color='black')   
    ls = LightSource(315, 45)

    rgb = ls.shade(crater_map, cmap=cmap, vmin=0, vmax=crater_map.max(), vert_exag=2, blend_mode='hsv')

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(rgb, cmap=cmap, vmin=0, vmax=crater_map.max(),
                extent=[0, scale * nc, 0, scale * nr], 
                interpolation='nearest', origin='upper')

    if solution:
        camino_x = [c * scale for _, c in solution]  
        camino_y = [(nr - r) * scale for r, _ in solution] 
        ax.plot(camino_x, camino_y, color='blue', linewidth=2, label="Camino recorrido")

    ax.scatter(x_start, y_start, color='green', marker='o', s=100, label="Inicio")
    ax.scatter(problem.goal[1] * scale, (nr - problem.goal[0]) * scale, color='red', marker='x', s=100, label="Meta")

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Altura (m)')

    plt.title('Cráter de Marte con camino recorrido')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend()

    plt.show()
class Marte_Problem:
    def __init__(self, initial, goal, maze):
        self.initial = initial
        self.goal = goal
        self.maze = maze
        self.rows, self.cols = maze.shape

    def goal_test(self, state):
        return state == self.goal

    def actions(self, state):
        x, y = state
        posibles_mov = [
            (x-1, y), 
            (x+1, y),   
            (x, y-1),   
            (x, y+1),   
            (x-1, y-1), 
            (x-1, y+1), 
            (x+1, y-1), 
            (x+1, y+1)  
        ]
        acciones_validas = []
        altura_actual = self.maze[x, y]
        for nx, ny in posibles_mov:
            if 0 <= nx < self.rows and 0 <= ny < self.cols and self.maze[nx, ny] != 255:
                altura_nueva = self.maze[nx, ny]
                if abs(altura_nueva - altura_actual) < 1.5:
                    acciones_validas.append((nx, ny))
        return acciones_validas



    def result(self, state, action):
        return action  
def camino_recorrido(mars_map, x_start, y_start, x_goal, y_goal):
    def calcrc(nr, nc, x, y, scale): 
        r = nr - round(y / scale)
        c = round(x / scale)
        return r, c


    scale = 10.0174
    nr, nc = mars_map.shape  

    start = calcrc(nr, nc, x_start, y_start, scale)
    goal = calcrc(nr, nc, x_goal, y_goal, scale)


    problem = Marte_Problem(start, goal, mars_map)
    gbfs = GreedyBestFirstSearch(problem)
    solution = gbfs.search()
    solution_scaled = [calxy(nr, nc, r, c, scale) for r, c in solution]
    print(f"La solución es: {solution_scaled}")
    print(f"Longitud del camino: {len(solution_scaled)}")
    print(f"Altura de la meta: {mars_map[goal]}")
    print(f"Altura del camino: {[mars_map[r, c] for r, c in solution]}")
    cmap = copy.copy(plt.cm.get_cmap('autumn'))
    cmap.set_under(color='black')   
    ls = LightSource(315, 45)


    rgb = ls.shade(mars_map, cmap=cmap, vmin=0, vmax=mars_map.max(), vert_exag=2, blend_mode='hsv')

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(rgb, cmap=cmap, vmin=0, vmax=mars_map.max(),
                extent=[0, scale * nc, 0, scale * nr], 
                interpolation='nearest', origin='upper')

    if solution:
        camino_x = [c * scale for _, c in solution]  
        camino_y = [(nr - r) * scale for r, _ in solution] 
        ax.plot(camino_x, camino_y, color='blue', linewidth=2, label="Camino recorrido")

    ax.scatter(x_start, y_start, color='green', marker='o', s=100, label="Inicio")
    ax.scatter(x_goal, y_goal, color='red', marker='x', s=100, label="Meta")

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Altura (m)')


    plt.title('Superficie de Marte con camino recorrido')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend()


    plt.show()