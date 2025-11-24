import pygame, random, math, numpy
from pygame.locals import *
from Cubo import Cubo
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import csv
import os
from datetime import datetime

# Variables globales 
NodosVisita = None
A = None
matrix_size = None
world_size = 100

def generar_matriz_adyacencia(filas, columnas):
    n = filas * columnas
    A = numpy.zeros((n, n), dtype=int)
    direcciones = [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for f in range(filas):
        for c in range(columnas):
            i = f * columnas + c
            for df, dc in direcciones:
                nf, nc = f + df, c + dc
                if 0 <= nf < filas and 0 <= nc < columnas:
                    j = nf * columnas + nc
                    A[i, j] = 1
    return A

def init_world(M):
    global NodosVisita, A, matrix_size
    if NodosVisita is not None:
        return  

    if M < 2:
        M = 2  # mínimo razonable

    matrix_size = M
    node_spacing = 20.0  # unidades entre nodos
    spacing = node_spacing

    # Calcular el offset para centrar la matriz en el origen (0, 0, 0)
    # El nodo central de la matriz (no el nodo 0) estará en (0, 0, 0)
    # Esto distribuye la matriz alrededor del centro del mapa
    center_offset = -(matrix_size - 1) / 2.0 * spacing

    nodos = []
    for i in range(matrix_size):
        for j in range(matrix_size):
            # Centrar la matriz: el nodo central estará en (0, 0, 0)
            # Para M=5: nodos van de -40 a +40, con el centro en 0
            x = j * spacing + center_offset
            z = i * spacing + center_offset
            nodos.append([x, 0, z])
    NodosVisita = numpy.asarray(nodos, dtype=numpy.float64)
    A = generar_matriz_adyacencia(matrix_size, matrix_size)

def planGen(N, A):
    """Genera rutas planeadas dinámicamente basándose en el tamaño de la matriz"""
    global matrix_size
    # Calcular M desde el tamaño de la matriz de adyacencia
    if matrix_size is not None:
        M = matrix_size
    elif A is not None and len(A) > 0:
        M = int(numpy.sqrt(len(A)))
    else:
        M = 5  # Valor por defecto
    total_nodes = M * M
    
    # Nodos importantes: esquinas y bordes
    top_left = 0
    top_right = M - 1
    bottom_left = M * (M - 1)
    bottom_right = M * M - 1
    
    match N:
        case 1:
            # Un solo agente: zigzag a través de toda la matriz
            route = []
            for i in range(M):
                if i % 2 == 0:
                    # De izquierda a derecha
                    for j in range(M):
                        route.append(i * M + j)
                else:
                    # De derecha a izquierda
                    for j in range(M - 1, -1, -1):
                        route.append(i * M + j)
            route.append(0)  # Volver al inicio
            return [route]
            
        case 2:
            # Dos agentes: dividir la matriz en dos zonas
            mid_point = M // 2
            route1 = []
            route2 = []
            
            # Ruta 1: parte superior
            for i in range(mid_point):
                if i % 2 == 0:
                    for j in range(M):
                        route1.append(i * M + j)
                else:
                    for j in range(M - 1, -1, -1):
                        route1.append(i * M + j)
            route1.append(0)
            
            # Ruta 2: parte inferior
            for i in range(mid_point, M):
                if (i - mid_point) % 2 == 0:
                    for j in range(M):
                        route2.append(i * M + j)
                else:
                    for j in range(M - 1, -1, -1):
                        route2.append(i * M + j)
            route2.append(0)
            
            return [route1, route2]
            
        case 3:
            # Tres agentes: dividir en tres zonas
            zone_size = M // 3
            route1 = []
            route2 = []
            route3 = []
            
            # Ruta 1: primera zona
            for i in range(zone_size):
                for j in range(M):
                    route1.append(i * M + j)
            route1.append(0)
            
            # Ruta 2: segunda zona
            for i in range(zone_size, 2 * zone_size):
                for j in range(M):
                    route2.append(i * M + j)
            route2.append(0)
            
            # Ruta 3: tercera zona
            for i in range(2 * zone_size, M):
                for j in range(M):
                    route3.append(i * M + j)
            route3.append(0)
            
            return [route1, route2, route3]
            
        case _:
            # Para más de 3 agentes, dividir la matriz en N zonas
            routes = []
            nodes_per_agent = total_nodes // N
            
            for agent_idx in range(N):
                route = []
                start_node = agent_idx * nodes_per_agent
                end_node = (agent_idx + 1) * nodes_per_agent if agent_idx < N - 1 else total_nodes
                
                for node in range(start_node, end_node):
                    route.append(node)
                route.append(0)  # Volver al inicio
                routes.append(route)
            
            return routes

def get_random_neighbor(current_node, A):
    """Obtiene un vecino aleatorio del nodo actual usando la matriz de adyacencia"""
    neighbors = []
    for i in range(len(A)):
        if A[current_node][i] == 1:
            neighbors.append(i)
    
    if neighbors:
        return random.choice(neighbors)
    else:
        return current_node  # Si no tiene vecinos, se queda en el mismo nodo
    
def ensure_data_directory():
    """Crea la carpeta datos si no existe"""
    if not os.path.exists('datos'):
        os.makedirs('datos')

def create_timestamp_filename(agent_id, method):
    """Crea el nombre del archivo con timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"datos/{timestamp}_agent{agent_id}_{method}.csv"

class Lifter:
    def __init__(self, dim, vel, textures, idx, position, currentNode, method="planned"):
        self.dim = dim
        self.idx = idx
        self.Position = position
        self.Direction = numpy.zeros(3)
        self.angle = 0
        self.vel = vel
        self.method = method  # "planned" o "random"
        
        # Inicializar las rutas PRIMERO según el método
        if self.method == "planned":
            self.original_route = planGen(idx + 1, A)[idx]
            self.route = self.original_route.copy()
            self.route_index = 0
        else:
            self.original_route = []
            self.route = []
            self.route_index = 0
        
        self.currentNode = currentNode
        self.nextNode = self.get_initial_next_node()
        
        # Arreglo de texturas
        self.textures = textures

        self.platformHeight = -1.5
        self.platformUp = False
        self.platformDown = False

        self.radiusCol = 5

        self.status = "searching"
        self.trashID = -1

        self.lastPickupNode = None
        self.returningToPickup = False
        self.finishedRoute = False
        self.finalNodeWithTrash = False
        self.workCompleted = False
        self.movements = 0 
        
        # Para método aleatorio: llevar cuenta de nodos visitados
        self.visited_nodes = set([currentNode])
        self.consecutive_repeats = 0
        self.max_consecutive_repeats = 3
        
        # Para detección de colisiones con otros lifters
        self.nearby_lifter_detected = False
        self.avoidance_cooldown = 0  # Tiempo de espera antes de poder cambiar dirección de nuevo
        self.drop_time = 0  # Tiempo en el área de tirar basura
        self.last_avoidance_node = None  # Último nodo al que cambió para evitar
        self.avoidance_count = 0  # Contador de evasiones consecutivas
        self.interference_count = 0  # Contador de interferencias/entorpecimientos (se incrementa en checkCollisions)
        self.temporary_avoidance_node = None  # Nodo temporal para evasión (planned)
        self.avoiding = False  # Flag para indicar que está evitando

        ensure_data_directory()  # Asegurar que la carpeta existe
        self.csv_file = create_timestamp_filename(idx, method)
        self.last_logged_state = None
        self.last_logged_node = None
        
        # Crear archivo CSV con headers
        self.init_csv()
        
    def init_csv(self):
        """Inicializa el archivo CSV con los headers"""
        with open(self.csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'timestamp', 'agent_id', 'method', 'state', 
                'pos_x', 'pos_y', 'pos_z', 
                'current_node', 'next_node', 'route'
            ])
    
    def log_to_csv(self, state_changed=False, node_changed=False):
        """Guarda datos en CSV solo cuando hay cambios significativos"""
        current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]  
        
        # Determinar si debemos guardar basado en cambios
        should_log = (
            state_changed or 
            node_changed or 
            self.last_logged_state is None or
            self.last_logged_node is None
        )
        
        if should_log:
            with open(self.csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                route_str = str(self.route) if self.method == "planned" else "random"
                
                writer.writerow([
                    current_time,
                    self.idx,
                    self.method,
                    self.status,
                    round(self.Position[0], 2),
                    round(self.Position[1], 2),
                    round(self.Position[2], 2),
                    self.currentNode,
                    self.nextNode,
                    route_str
                ])
            
            self.last_logged_state = self.status
            self.last_logged_node = self.currentNode

    def get_initial_next_node(self):
        """Obtiene el siguiente nodo inicial según el método"""
        if self.method == "planned":
            if len(self.original_route) > 1:
                return self.original_route[1]
            else:
                return self.currentNode
        else:
            # Para random, elegir un vecino aleatorio diferente
            next_node = get_random_neighbor(self.currentNode, A)
            # Si es el nodo 0 y hay múltiples vecinos, preferir uno diferente
            if self.currentNode == 0:
                neighbors = []
                for i in range(len(A)):
                    if A[self.currentNode][i] == 1 and i != self.currentNode:
                        neighbors.append(i)
                if len(neighbors) > 1:
                    # Preferir un nodo que no sea el 0
                    non_zero_neighbors = [n for n in neighbors if n != 0]
                    if non_zero_neighbors:
                        next_node = random.choice(non_zero_neighbors)
            return next_node

    def search(self):
        u = numpy.random.rand(3)
        u[1] = 0
        u /= numpy.linalg.norm(u)
        self.Direction = u

    def targetCenter(self):
        # Set direction to center
        dirX = -self.Position[0]
        dirZ = -self.Position[2]
        magnitude = math.sqrt(dirX**2 + dirZ**2)
        self.Direction = [(dirX / magnitude), 0, (dirZ / magnitude)]

    def ComputeDirection(self, Posicion, NodoSiguiente):
        Direccion = NodosVisita[NodoSiguiente,:] - Posicion
        Direccion = numpy.asarray(Direccion)
        Distancia = numpy.linalg.norm(Direccion)
        Direccion /= Distancia
        return Direccion, Distancia

    def isFinalNode(self, node):
        """Verifica si es un nodo final importante de la ruta (esquinas y bordes)"""
        global matrix_size, A
        # Calcular M desde el tamaño de la matriz de adyacencia
        if matrix_size is not None:
            M = matrix_size
        elif A is not None and len(A) > 0:
            M = int(numpy.sqrt(len(A)))
        else:
            M = 5  # Valor por defecto
        total_nodes = M * M
        
        # Nodos finales: esquinas y nodos del borde derecho
        final_nodes = [
            0,  # top-left
            M - 1,  # top-right
            M * (M - 1),  # bottom-left
            M * M - 1  # bottom-right
        ]
        
        # Agregar nodos del borde derecho (columna final)
        for i in range(1, M - 1):
            final_nodes.append(i * M + (M - 1))
        
        return node in final_nodes
    
    def isInTrashArea(self):
        """Verifica si el lifter está en el área de tirar basura"""
        return (self.Position[0] <= 10 and self.Position[0] >= -10) and \
               (self.Position[2] <= 10 and self.Position[2] >= -10)
    
    def avoidLifter(self, other_lifter_position):
        """Cambia la dirección para evitar otro lifter (solo para random)"""
        if self.method != "random" or self.isInTrashArea():
            return False
        
        # Calcular dirección de escape
        escape_dir = self.Position - other_lifter_position
        escape_dir[1] = 0  # Mantener en el plano horizontal
        distance = numpy.linalg.norm(escape_dir)
        
        if distance > 0:
            escape_dir /= distance
            # Cambiar a un nodo aleatorio diferente
            neighbors = []
            for i in range(len(A)):
                if A[self.currentNode][i] == 1 and i != self.currentNode:
                    neighbors.append(i)
            if neighbors:
                self.nextNode = random.choice(neighbors)
                return True
        return False

    def RetrieveNextNodePath(self, NodoActual):
        """Obtiene el siguiente nodo según el método de navegación"""
        
        if self.workCompleted:
            return self.currentNode
        
        # Si está evitando y tiene un nodo temporal, usarlo primero
        if self.method == "planned" and self.avoiding and self.temporary_avoidance_node is not None:
            # Si ya llegó al nodo temporal, volver a la ruta normal
            if self.currentNode == self.temporary_avoidance_node:
                self.avoiding = False
                self.temporary_avoidance_node = None
                return self._get_next_planned_node(NodoActual)
            else:
                # Seguir hacia el nodo temporal
                return self.temporary_avoidance_node
        
        if self.method == "planned":
            return self._get_next_planned_node(NodoActual)
        else:
            return self._get_next_random_node(NodoActual)

    def _get_next_planned_node(self, NodoActual):
        """Lógica para método planeado"""
        self.route_index += 1
        
        if self.route_index >= len(self.route):
            if not self.finishedRoute:
                self.finishedRoute = True
                print(f"El agente {self.idx} completó su ruta planeada.")
                
                if self.finalNodeWithTrash:
                    print(f"El agente {self.idx} encontró basura en el nodo final, realizando verificación final.")
                    final_node = self.route[-2] if self.route[-1] == 0 else self.route[-1]
                    self.route = [0, final_node, 0]
                    self.route_index = 0
                    self.finalNodeWithTrash = False
                    return self.route[0]
                else:
                    print(f"El agente {self.idx} terminó su trabajo - regresando al nodo 0")
                    self.workCompleted = True
                    return 0
            
            else:
                print(f"El agente {self.idx} terminó todo el trabajo")
                self.workCompleted = True
                return 0
            
        next_node = self.route[self.route_index]
        return next_node

    def _get_next_random_node(self, NodoActual):
        """Lógica para método aleatorio"""
        if self.returningToPickup:
            print(f"El agente {self.idx} regresó al nodo de recolección {self.lastPickupNode}.")
            self.returningToPickup = False
            self.lastPickupNode = None
            # Continuar exploración aleatoria
            return get_random_neighbor(NodoActual, A)
        
        # Elegir un vecino aleatorio
        next_node = get_random_neighbor(NodoActual, A)
        
        # Evitar repetir el mismo nodo muchas veces consecutivas
        if next_node == self.currentNode:
            self.consecutive_repeats += 1
            if self.consecutive_repeats >= self.max_consecutive_repeats:
                # Forzar cambio a un nodo diferente
                neighbors = []
                for i in range(len(A)):
                    if A[NodoActual][i] == 1 and i != NodoActual:
                        neighbors.append(i)
                if neighbors:
                    next_node = random.choice(neighbors)
                    self.consecutive_repeats = 0
        else:
            self.consecutive_repeats = 0
        
        # Registrar nodo visitado
        self.visited_nodes.add(next_node)
        
        # Ocasionalmente volver al nodo 0 para "reiniciar" la exploración
        if random.random() < 0.05 and NodoActual != 0:  
            print(f"El agente {self.idx} regresó aleatoriamente al nodo 0")
            return 0
            
        return next_node

    def update(self, delta):
        # Guardar estado y nodo anteriores para detectar cambios
        previous_state = self.status
        previous_node = self.currentNode
        
        # Reducir cooldown de evasión
        if self.avoidance_cooldown > 0:
            self.avoidance_cooldown -= delta
        
        if self.workCompleted:
            if random.random() < 0.01:
                print(f"El agente {self.idx} - Trabajo completado. Esperando en el nodo 0.")
            return
        
        # Si detectó otro lifter cercano (random o planned), cambiar dirección de forma más agresiva
        if self.nearby_lifter_detected and self.avoidance_cooldown <= 0:
            if not self.isInTrashArea() and self.status == "searching":
                if self.method == "random":
                    # Para random: cambiar a un nodo aleatorio diferente
                    neighbors = []
                    for i in range(len(A)):
                        if A[self.currentNode][i] == 1 and i != self.currentNode:
                            # Evitar el siguiente nodo actual y el último nodo de evasión si es muy reciente
                            if i != self.nextNode and (i != self.last_avoidance_node or self.avoidance_count < 3):
                                neighbors.append(i)
                    
                    if neighbors:
                        # Preferir nodos que no hayan sido visitados recientemente (últimos 5 nodos)
                        unvisited_neighbors = [n for n in neighbors if n not in list(self.visited_nodes)[-5:]]
                        if unvisited_neighbors:
                            self.nextNode = random.choice(unvisited_neighbors)
                        else:
                            # Si todos fueron visitados, elegir aleatoriamente
                            self.nextNode = random.choice(neighbors)
                        
                        self.last_avoidance_node = self.nextNode
                        self.avoidance_count += 1
                        # Incrementar interferencia solo cuando realmente cambia de dirección
                        self.interference_count += 1
                        self.avoidance_cooldown = 0.15  # Cooldown más corto para respuesta más rápida
                        print(f"El agente {self.idx} (random) detectó otro lifter, cambiando dirección al nodo {self.nextNode}")
                    else:
                        # Si no hay vecinos disponibles, forzar cambio de dirección inmediato
                        # Cambiar la dirección actual para alejarse
                        self.Direction[0] *= -1
                        self.Direction[2] *= -1
                        self.avoidance_cooldown = 0.1
                elif self.method == "planned":
                    # Para planned: desviarse temporalmente a un nodo vecino que no esté en la ruta inmediata
                    neighbors = []
                    for i in range(len(A)):
                        if A[self.currentNode][i] == 1 and i != self.currentNode:
                            # Preferir nodos que no sean el siguiente en la ruta
                            if i != self.nextNode:
                                neighbors.append(i)
                    
                    if neighbors:
                        # Elegir un vecino que esté en dirección opuesta o perpendicular
                        # Calcular dirección hacia el siguiente nodo en la ruta
                        if self.nextNode < len(NodosVisita):
                            route_dir = NodosVisita[self.nextNode] - self.Position
                            route_dir[1] = 0
                            route_dir /= numpy.linalg.norm(route_dir) if numpy.linalg.norm(route_dir) > 0 else 1
                            
                            # Elegir el vecino que esté más en dirección opuesta
                            best_neighbor = neighbors[0]
                            max_dot = -1
                            for n in neighbors:
                                if n < len(NodosVisita):
                                    neighbor_dir = NodosVisita[n] - self.Position
                                    neighbor_dir[1] = 0
                                    neighbor_dir /= numpy.linalg.norm(neighbor_dir) if numpy.linalg.norm(neighbor_dir) > 0 else 1
                                    dot = numpy.dot(route_dir, neighbor_dir)
                                    if dot < max_dot:
                                        max_dot = dot
                                        best_neighbor = n
                            
                            self.temporary_avoidance_node = best_neighbor
                            self.nextNode = best_neighbor
                            self.avoiding = True
                            self.interference_count += 1
                            self.avoidance_cooldown = 0.3
                            print(f"El agente {self.idx} (planned) detectó otro lifter, desviándose temporalmente al nodo {self.nextNode}")
                        else:
                            # Si no hay vecinos, cambiar dirección temporalmente
                            self.Direction[0] *= -1
                            self.Direction[2] *= -1
                            self.avoidance_cooldown = 0.1
                    else:
                        # Si no hay vecinos, cambiar dirección temporalmente
                        self.Direction[0] *= -1
                        self.Direction[2] *= -1
                        self.avoidance_cooldown = 0.1
            self.nearby_lifter_detected = False
        
        Direccion, Distancia = self.ComputeDirection(self.Position, self.nextNode)
        
        if Distancia < 2:
            print(f"El agente {self.idx} está calculando un nuevo nodo...")
            self.currentNode = self.nextNode
            self.movements += 1
            # Resetear contador de evasiones cuando llega a un nodo
            if self.avoidance_count > 0:
                self.avoidance_count = 0
            
            # Si estaba evitando y llegó al nodo temporal, resetear el flag
            if self.avoiding and self.temporary_avoidance_node == self.currentNode:
                self.avoiding = False
                self.temporary_avoidance_node = None

            if self.returningToPickup:
                print(f"El agente {self.idx} regresó al nodo de recolección {self.lastPickupNode}.")
                self.returningToPickup = False
                self.lastPickupNode = None
                self.nextNode = self.RetrieveNextNodePath(self.currentNode)
            else:
                self.nextNode = self.RetrieveNextNodePath(self.currentNode)

        mssg = "Agent:%d \t Method:%s \t State:%s \t Position:[%0.2f,0,%0.2f] \t NodoActual:%d \t NodoSiguiente:%d" % (
            self.idx, self.method, self.status, self.Position[0], self.Position[-1], self.currentNode, self.nextNode
        )
        if self.method == "planned":
            mssg += f" \t Route:{self.route}"
        print(mssg)

        match self.status:
            case "searching":
                # Verificar colisiones durante el movimiento (para random y planned)
                if self.nearby_lifter_detected and not self.isInTrashArea():
                    # Si detecta otro lifter durante el movimiento, ajustar dirección más agresivamente
                    if self.avoidance_cooldown <= 0:
                        # Aplicar un ajuste más fuerte de dirección perpendicular para evitar
                        perpendicular_dir = numpy.array([-Direccion[2], 0, Direccion[0]])
                        # Aumentar el factor de evasión para mantener más distancia
                        Direccion = (Direccion + perpendicular_dir * 0.6)
                        Direccion /= numpy.linalg.norm(Direccion)
                        self.avoidance_cooldown = 0.1
                
                self.Position += Direccion * self.vel
                self.Direction = Direccion
                self.angle = math.acos(self.Direction[0]) * 180 / math.pi
                if self.Direction[2] > 0:
                    self.angle = 360 - self.angle

                if self.platformUp:
                    if self.platformHeight >= 0:
                        self.platformUp = False
                    else:
                        self.platformHeight += delta
                elif self.platformDown:
                    if self.platformHeight <= -1.5:
                        self.platformUp = True
                    else:
                        self.platformHeight -= delta
            case "lifting":
                if self.platformHeight >= 0:
                    self.targetCenter()
                    self.status = "delivering"
                else:
                    self.platformHeight += delta
            case "delivering":
                if (self.Position[0] <= 10 and self.Position[0] >= -10) and (self.Position[2] <= 10 and self.Position[2] >= -10):
                    self.status = "dropping"
                else:
                    newX = self.Position[0] + self.Direction[0] * self.vel
                    newZ = self.Position[2] + self.Direction[2] * self.vel
                    if newX - 10 < -self.dim or newX + 10 > self.dim:
                        self.Direction[0] *= -1
                    else:
                        self.Position[0] = newX
                    if newZ - 10 < -self.dim or newZ + 10 > self.dim:
                        self.Direction[2] *= -1
                    else:
                        self.Position[2] = newZ
                    self.angle = math.acos(self.Direction[0]) * 180 / math.pi
                    if self.Direction[2] > 0:
                        self.angle = 360 - self.angle
            case "dropping":
                # Si es planned, reducir tiempo de detención (bajar más rápido)
                drop_speed = delta * 2.0 if self.method == "planned" else delta
                
                if self.platformHeight <= -1.5:
                    if self.lastPickupNode is not None:
                        if self.isFinalNode(self.lastPickupNode):
                            self.finalNodeWithTrash = True
                            print(f"El agente {self.idx} encontró basura en el nodo final {self.lastPickupNode}")
                        
                        self.nextNode = self.lastPickupNode
                        self.returningToPickup = True
                        print(f"El agente {self.idx} regresó al nodo {self.lastPickupNode} para buscar más basura")
                    self.status = "searching"
                    self.drop_time = 0
                else:
                    self.platformHeight -= drop_speed
                    self.drop_time += delta

            case "returning":
                if (self.Position[0] <= 20 and self.Position[0] >= -20) and (self.Position[2] <= 20 and self.Position[2] >= -20):
                    self.Position[0] -= (self.Direction[0] * (self.vel/4))
                    self.Position[2] -= (self.Direction[2] * (self.vel/4))
                else:
                    self.search()
                    self.status = "searching"

        state_changed = (previous_state != self.status)
        node_changed = (previous_node != self.currentNode)
        
        # Loggear solo si hay cambios
        self.log_to_csv(state_changed, node_changed)

    def draw(self):
        glPushMatrix()
        glTranslatef(self.Position[0], self.Position[1], self.Position[2])
        glRotatef(self.angle, 0, 1, 0)
        glScaled(5, 5, 5)
        glColor3f(1.0, 0.7, 0.9)
        # front face
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.textures[2])
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0)
        glVertex3d(1, 1, 1)
        glTexCoord2f(0.0, 1.0)
        glVertex3d(1, 1, -1)
        glTexCoord2f(1.0, 1.0)
        glVertex3d(1, -1, -1)
        glTexCoord2f(1.0, 0.0)
        glVertex3d(1, -1, 1)

        # 2nd face
        glTexCoord2f(0.0, 0.0)
        glVertex3d(-2, 1, 1)
        glTexCoord2f(0.0, 1.0)
        glVertex3d(1, 1, 1)
        glTexCoord2f(1.0, 1.0)
        glVertex3d(1, -1, 1)
        glTexCoord2f(1.0, 0.0)
        glVertex3d(-2, -1, 1)

        # 3rd face
        glTexCoord2f(0.0, 0.0)
        glVertex3d(-2, 1, -1)
        glTexCoord2f(0.0, 1.0)
        glVertex3d(-2, 1, 1)
        glTexCoord2f(1.0, 1.0)
        glVertex3d(-2, -1, 1)
        glTexCoord2f(1.0, 0.0)
        glVertex3d(-2, -1, -1)

        # 4th face
        glTexCoord2f(0.0, 0.0)
        glVertex3d(1, 1, -1)
        glTexCoord2f(0.0, 1.0)
        glVertex3d(-2, 1, -1)
        glTexCoord2f(1.0, 1.0)
        glVertex3d(-2, -1, -1)
        glTexCoord2f(1.0, 0.0)
        glVertex3d(1, -1, -1)

        # top
        glTexCoord2f(0.0, 0.0)
        glVertex3d(1, 1, 1)
        glTexCoord2f(0.0, 1.0)
        glVertex3d(-2, 1, 1)
        glTexCoord2f(1.0, 1.0)
        glVertex3d(-2, 1, -1)
        glTexCoord2f(1.0, 0.0)
        glVertex3d(1, 1, -1)
        glEnd()

        # Head
        glPushMatrix()
        glTranslatef(0, 1.5, 0)
        glScaled(0.8, 0.8, 0.8)
        glColor3f(1.0, 0.7, 0.9)
        head = Cubo(self.textures, 0)
        head.draw()
        glPopMatrix()
        glDisable(GL_TEXTURE_2D)

        # Wheels 
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.textures[1])
        glPushMatrix()
        glTranslatef(-1.2, -1, 1)
        glScaled(0.3, 0.3, 0.3)
        glColor3f(1.0, 0.7, 0.9)
        wheel = Cubo(self.textures, 0)
        wheel.draw()
        glPopMatrix()

        glPushMatrix()
        glTranslatef(0.5, -1, 1)
        glScaled(0.3, 0.3, 0.3)
        wheel = Cubo(self.textures, 0)
        wheel.draw()
        glPopMatrix()

        glPushMatrix()
        glTranslatef(0.5, -1, -1)
        glScaled(0.3, 0.3, 0.3)
        wheel = Cubo(self.textures, 0)
        wheel.draw()
        glPopMatrix()

        glPushMatrix()
        glTranslatef(-1.2, -1, -1)
        glScaled(0.3, 0.3, 0.3)
        wheel = Cubo(self.textures, 0)
        wheel.draw()
        glPopMatrix()
        glDisable(GL_TEXTURE_2D)

        # Lifter
        glPushMatrix()
        if self.status in ["lifting","delivering","dropping"]:
            self.drawTrash()
        glColor3f(0.0, 0.0, 0.0)
        glTranslatef(0, self.platformHeight, 0)  # Up and down
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0)
        glVertex3d(1, 1, 1)
        glTexCoord2f(0.0, 1.0)
        glVertex3d(1, 1, -1)
        glTexCoord2f(1.0, 1.0)
        glVertex3d(3, 1, -1)
        glTexCoord2f(1.0, 0.0)
        glVertex3d(3, 1, 1)
        glEnd()
        glPopMatrix()
        glPopMatrix()

    def drawTrash(self):
        glPushMatrix()
        glTranslatef(2, (self.platformHeight + 1.5), 0)
        glScaled(0.5, 0.5, 0.5)
        glColor3f(1.0, 1.0, 1.0)

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.textures[3])

        glBegin(GL_QUADS)
        # Front face
        glTexCoord2f(0.0, 0.0)
        glVertex3d(1, 1, 1)
        glTexCoord2f(1.0, 0.0)
        glVertex3d(-1, 1, 1)
        glTexCoord2f(1.0, 1.0)
        glVertex3d(-1, -1, 1)
        glTexCoord2f(0.0, 1.0)
        glVertex3d(1, -1, 1)

        # Back face
        glTexCoord2f(0.0, 0.0)
        glVertex3d(-1, 1, -1)
        glTexCoord2f(1.0, 0.0)
        glVertex3d(1, 1, -1)
        glTexCoord2f(1.0, 1.0)
        glVertex3d(1, -1, -1)
        glTexCoord2f(0.0, 1.0)
        glVertex3d(-1, -1, -1)

        # Left face
        glTexCoord2f(0.0, 0.0)
        glVertex3d(-1, 1, 1)
        glTexCoord2f(1.0, 0.0)
        glVertex3d(-1, 1, -1)
        glTexCoord2f(1.0, 1.0)
        glVertex3d(-1, -1, -1)
        glTexCoord2f(0.0, 1.0)
        glVertex3d(-1, -1, 1)

        # Right face
        glTexCoord2f(0.0, 0.0)
        glVertex3d(1, 1, -1)
        glTexCoord2f(1.0, 0.0)
        glVertex3d(1, 1, 1)
        glTexCoord2f(1.0, 1.0)
        glVertex3d(1, -1, 1)
        glTexCoord2f(0.0, 1.0)
        glVertex3d(1, -1, -1)

        # Top face
        glTexCoord2f(0.0, 0.0)
        glVertex3d(-1, 1, 1)
        glTexCoord2f(1.0, 0.0)
        glVertex3d(1, 1, 1)
        glTexCoord2f(1.0, 1.0)
        glVertex3d(1, 1, -1)
        glTexCoord2f(0.0, 1.0)
        glVertex3d(-1, 1, -1)

        # Bottom face
        glTexCoord2f(0.0, 0.0)
        glVertex3d(-1, -1, 1)
        glTexCoord2f(1.0, 0.0)
        glVertex3d(1, -1, 1)
        glTexCoord2f(1.0, 1.0)
        glVertex3d(1, -1, -1)
        glTexCoord2f(0.0, 1.0)
        glVertex3d(-1, -1, -1)
        glEnd()
        glDisable(GL_TEXTURE_2D)

        glPopMatrix()