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

    # Permitir cualquier valor positivo de M
    if M < 1:
        M = 1  # mínimo absoluto (solo un nodo)
    
    matrix_size = M
    # Usar DimBoard para distribuir los nodos en el tablero
    # El tablero va de -DimBoard a +DimBoard, así que el rango total es 2*DimBoard
    DimBoard = 200  # Valor del tablero
    
    # Calcular espaciado entre nodos basado en el tamaño de la matriz
    # Con más nodos (M mayor), el espaciado es menor
    # Con menos nodos (M menor), el espaciado es mayor
    if matrix_size > 1:
        node_spacing = (2 * DimBoard) / (matrix_size - 1)
    else:
        node_spacing = 2 * DimBoard
    
    # Empezar desde -DimBoard (esquina) hasta +DimBoard
    start_pos = -DimBoard
    nodos = []
    for i in range(matrix_size):
        for j in range(matrix_size):
            x = start_pos + j * node_spacing
            z = start_pos + i * node_spacing
            nodos.append([x, 0, z])
    
    NodosVisita = numpy.asarray(nodos, dtype=numpy.float64)
    A = generar_matriz_adyacencia(matrix_size, matrix_size)
    
    print(f"Matriz inicializada: {matrix_size}x{matrix_size} nodos")
    print(f"Espaciado entre nodos: {node_spacing:.2f} unidades")
    print(f"Rango: de {-DimBoard} a {DimBoard} en X y Z")

def planGen(N, A):
    """
    Genera rutas planeadas para N agentes basándose en el tamaño de la matriz.
    Las rutas se adaptan automáticamente al tamaño de la matriz.
    """
    global matrix_size
    if matrix_size is None:
        matrix_size = 5  # Default si no está inicializado
    
    total_nodes = matrix_size * matrix_size
    
    # Si la matriz es 5x5, usar las rutas optimizadas originales
    if matrix_size == 5:
        match N:
            case 1:
                # Un solo agente: zigzag 
                return [[0, 1, 2, 3, 4, 9, 8, 7, 6, 5, 10, 11, 12, 13, 14, 19, 18, 17, 16, 15, 20, 21, 22, 23, 24, 0]]
            case 2:
                # Dos agentes
                return [
                    [0, 1, 2, 3, 4, 9, 14, 19, 24, 23, 18, 13, 8, 7, 12, 0],  # Ruta 1
                    [5, 10, 15, 20, 21, 22, 17, 16, 11, 6, 0]  # Ruta 2
                ]
            case 3:
                # Tres agentes
                return [
                    [0, 1, 2, 3, 4, 9, 14, 19, 24, 23, 18, 13, 8, 0],  # Ruta 1
                    [5, 10, 15, 20, 21, 22, 17, 12, 7, 0],  # Ruta 2
                    [0, 5, 6, 11, 16, 0]  # Ruta 3
                ]
    
    # Para otros tamaños de matriz, generar rutas adaptativas
    routes = []
    
    if N == 1:
        # Un solo agente: recorre todos los nodos en zigzag
        route = list(range(total_nodes))
        route.append(0)  # Volver al inicio
        routes.append(route)
    elif N == 2:
        # Dos agentes: dividir el mapa en dos mitades
        mid_point = total_nodes // 2
        # Primera mitad
        route1 = list(range(0, mid_point))
        route1.append(0)
        routes.append(route1)
        # Segunda mitad
        route2 = list(range(mid_point, total_nodes))
        route2.append(0)
        routes.append(route2)
    else:
        # Para 3 o más agentes: dividir el mapa en N secciones
        nodes_per_agent = total_nodes // N
        for i in range(N):
            start = i * nodes_per_agent
            end = start + nodes_per_agent if i < N - 1 else total_nodes
            route = list(range(start, end))
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
    def __init__(self, dim, vel, textures, idx, position, currentNode, method="planned", area_limit=None):
        self.dim = dim
        self.idx = idx
        self.Position = position
        self.area_limit = area_limit  # Límites del área de simulación (min, max)
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
        self.collisionAvoidanceRadius = 15  # Radio para detectar otros lifters y evitar colisiones

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
        self.last_direction = None  # Para suavizar cambios de dirección en método aleatorio
        self.node_change_cooldown = 0  # Cooldown para evitar cambios muy frecuentes de nodo

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
                next_node = self.original_route[1]
                # Asegurar que no sea el mismo nodo
                if next_node == self.currentNode and len(self.original_route) > 2:
                    next_node = self.original_route[2]
                elif next_node == self.currentNode:
                    # Si aún es el mismo, buscar un vecino
                    neighbors = []
                    for i in range(len(A)):
                        if A[self.currentNode][i] == 1 and i != self.currentNode:
                            neighbors.append(i)
                    if neighbors:
                        next_node = random.choice(neighbors)
                return next_node
            else:
                # Si la ruta tiene solo un nodo, buscar un vecino
                neighbors = []
                for i in range(len(A)):
                    if A[self.currentNode][i] == 1 and i != self.currentNode:
                        neighbors.append(i)
                if neighbors:
                    return random.choice(neighbors)
                return self.currentNode
        else:
            next_node = get_random_neighbor(self.currentNode, A)
            # Asegurar que no sea el mismo nodo
            if next_node == self.currentNode:
                neighbors = []
                for i in range(len(A)):
                    if A[self.currentNode][i] == 1 and i != self.currentNode:
                        neighbors.append(i)
                if neighbors:
                    next_node = random.choice(neighbors)
            return next_node

    def search(self):
        u = numpy.random.rand(3)
        u[1] = 0
        norm = numpy.linalg.norm(u)
        if norm > 0:
            u /= norm
        self.Direction = u

    def targetCenter(self):
        # Set direction to center
        dirX = -self.Position[0]
        dirZ = -self.Position[2]
        magnitude = math.sqrt(dirX**2 + dirZ**2)
        if magnitude > 0:
            self.Direction = numpy.array([(dirX / magnitude), 0, (dirZ / magnitude)])
        else:
            self.Direction = numpy.zeros(3)

    def ComputeDirection(self, Posicion, NodoSiguiente):
        Direccion = NodosVisita[NodoSiguiente,:] - Posicion
        Direccion = numpy.asarray(Direccion)
        Distancia = numpy.linalg.norm(Direccion)
        if Distancia > 0:
            Direccion /= Distancia
        return Direccion, Distancia
    
    def checkLifterCollisions(self, other_lifters):
        """
        Detecta otros lifters cercanos y calcula un vector de evitación.
        NO aplica evitación si ambos lifters están en el área del incinerador (centro).
        
        Args:
            other_lifters: Lista de otros objetos Lifter a verificar
            
        Returns:
            Vector de evitación normalizado, o None si no hay colisiones inminentes
        """
        # Área del incinerador (centro) - permitir colisiones aquí
        incinerator_radius = 25  # Radio del área donde se permite colisionar
        
        # Verificar si este lifter está en el área del incinerador
        distance_to_center = math.sqrt(self.Position[0]**2 + self.Position[2]**2)
        self_in_incinerator = distance_to_center <= incinerator_radius
        
        avoidance_vector = numpy.zeros(3)
        collision_detected = False
        
        for other in other_lifters:
            if other.idx == self.idx or other.workCompleted:
                continue
                
            # Verificar si el otro lifter está en el área del incinerador
            other_distance_to_center = math.sqrt(other.Position[0]**2 + other.Position[2]**2)
            other_in_incinerator = other_distance_to_center <= incinerator_radius
            
            # Si ambos están en el incinerador, permitir colisión (no evitar)
            if self_in_incinerator and other_in_incinerator:
                continue
                
            # Calcular distancia en el plano XZ (ignorar Y)
            dx = other.Position[0] - self.Position[0]
            dz = other.Position[2] - self.Position[2]
            distance = math.sqrt(dx * dx + dz * dz)
            
            # Si está dentro del radio de evitación
            if distance < self.collisionAvoidanceRadius and distance > 0:
                collision_detected = True
                # Calcular vector de repulsión (alejarse del otro lifter)
                repulsion_strength = (self.collisionAvoidanceRadius - distance) / self.collisionAvoidanceRadius
                avoidance_vector[0] -= (dx / distance) * repulsion_strength
                avoidance_vector[2] -= (dz / distance) * repulsion_strength
        
        if collision_detected:
            # Normalizar el vector de evitación
            magnitude = numpy.linalg.norm(avoidance_vector)
            if magnitude > 0:
                avoidance_vector /= magnitude
                return avoidance_vector
        
        return None
    
    def applyCollisionAvoidance(self, original_direction, other_lifters):
        """
        Aplica evitación de colisiones modificando la dirección original.
        
        Args:
            original_direction: Dirección original hacia el siguiente nodo (puede ser lista o numpy array)
            other_lifters: Lista de otros lifters
            
        Returns:
            Dirección modificada que evita colisiones (numpy array)
        """
        # Asegurar que original_direction sea un numpy array
        if not isinstance(original_direction, numpy.ndarray):
            original_direction = numpy.asarray(original_direction)
        
        avoidance = self.checkLifterCollisions(other_lifters)
        
        if avoidance is not None:
            # Combinar dirección original con vector de evitación
            # Peso mayor a la evitación cuando está muy cerca
            avoidance_weight = 0.6
            direction_weight = 0.4
            
            combined_direction = (original_direction * direction_weight + 
                                avoidance * avoidance_weight)
            
            # Normalizar
            magnitude = numpy.linalg.norm(combined_direction)
            if magnitude > 0:
                combined_direction /= magnitude
                return combined_direction
        
        return original_direction

    def isFinalNode(self, node):
        """Verifica si es un nodo final importante de la ruta"""
        final_nodes = [24, 4, 9, 14, 19]
        return node in final_nodes
    
    def _find_alternative_node_within_area(self):
        """Encuentra un nodo alternativo dentro del área permitida"""
        if self.area_limit is None:
            return
        
        # Buscar nodos dentro del área
        valid_nodes = []
        for i, node_pos in enumerate(NodosVisita):
            if (self.area_limit['min'] <= node_pos[0] <= self.area_limit['max'] and
                self.area_limit['min'] <= node_pos[2] <= self.area_limit['max']):
                valid_nodes.append(i)
        
        if valid_nodes:
            # Elegir el nodo válido más cercano a la posición actual
            distances = [numpy.linalg.norm(NodosVisita[n] - self.Position) for n in valid_nodes]
            closest_idx = numpy.argmin(distances)
            self.nextNode = valid_nodes[closest_idx]
            print(f"Agente {self.idx}: Nodo {self.nextNode} fuera del área, cambiando a nodo {self.nextNode} dentro del área")
        else:
            # Si no hay nodos válidos, usar el nodo actual
            self.nextNode = self.currentNode

    def RetrieveNextNodePath(self, NodoActual):
        """Obtiene el siguiente nodo según el método de navegación"""
        
        if self.workCompleted:
            return self.currentNode
        
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
                    next_node = self.route[0]
                    # Asegurar que no sea el mismo nodo
                    if next_node == NodoActual and len(self.route) > 1:
                        next_node = self.route[1]
                    return next_node
                else:
                    print(f"El agente {self.idx} terminó su trabajo - regresando al nodo 0")
                    self.workCompleted = True
                    return 0
            
            else:
                print(f"El agente {self.idx} terminó todo el trabajo")
                self.workCompleted = True
                return 0
            
        next_node = self.route[self.route_index]
        
        # Asegurar que no sea el mismo nodo que el actual
        if next_node == NodoActual:
            # Buscar el siguiente nodo en la ruta que sea diferente
            for i in range(self.route_index + 1, len(self.route)):
                if self.route[i] != NodoActual:
                    self.route_index = i
                    next_node = self.route[i]
                    break
            # Si aún es el mismo, buscar un vecino
            if next_node == NodoActual:
                neighbors = []
                for i in range(len(A)):
                    if A[NodoActual][i] == 1 and i != NodoActual:
                        neighbors.append(i)
                if neighbors:
                    next_node = random.choice(neighbors)
                    print(f"El agente {self.idx}: Evitando nodo repetido en ruta planeada, usando vecino {next_node}")
        
        return next_node

    def _get_next_random_node(self, NodoActual):
        """Lógica para método aleatorio con suavizado de dirección"""
        if self.returningToPickup:
            print(f"El agente {self.idx} regresó al nodo de recolección {self.lastPickupNode}.")
            self.returningToPickup = False
            self.lastPickupNode = None
            # Continuar exploración aleatoria
            next_node = get_random_neighbor(NodoActual, A)
            # Asegurar que no sea el mismo nodo
            if next_node == NodoActual:
                neighbors = []
                for i in range(len(A)):
                    if A[NodoActual][i] == 1 and i != NodoActual:
                        neighbors.append(i)
                if neighbors:
                    next_node = random.choice(neighbors)
            return next_node
        
        # Obtener todos los vecinos válidos
        neighbors = []
        for i in range(len(A)):
            if A[NodoActual][i] == 1 and i != NodoActual:
                neighbors.append(i)
        
        if not neighbors:
            print(f"El agente {self.idx}: Advertencia: No hay vecinos disponibles en nodo {NodoActual}")
            return NodoActual
        
        # Si hay una dirección anterior, preferir nodos en direcciones similares
        if self.last_direction is not None and len(neighbors) > 1:
            # Calcular dirección hacia cada vecino
            neighbor_scores = []
            current_pos = NodosVisita[NodoActual]
            
            for neighbor_idx in neighbors:
                neighbor_pos = NodosVisita[neighbor_idx]
                direction_to_neighbor = neighbor_pos - current_pos
                direction_to_neighbor[1] = 0  # Ignorar componente Y
                norm = numpy.linalg.norm(direction_to_neighbor)
                if norm > 0:
                    direction_to_neighbor /= norm
                    
                    # Calcular similitud con la dirección anterior (producto punto)
                    similarity = numpy.dot(self.last_direction, direction_to_neighbor)
                    neighbor_scores.append((neighbor_idx, similarity))
                else:
                    neighbor_scores.append((neighbor_idx, -1.0))
            
            # Ordenar por similitud (mayor es mejor)
            neighbor_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Preferir nodos con similitud > 0.3 (ángulo < ~72 grados), pero ocasionalmente elegir uno diferente
            if random.random() < 0.7 and neighbor_scores[0][1] > 0.3:
                # 70% de probabilidad de elegir un nodo en dirección similar
                # Elegir entre los top 2 más similares
                top_candidates = [n[0] for n in neighbor_scores[:2] if n[1] > 0.3]
                if top_candidates:
                    next_node = random.choice(top_candidates)
                else:
                    next_node = neighbor_scores[0][0]
            else:
                # 30% de probabilidad de explorar en otra dirección
                next_node = random.choice(neighbors)
        else:
            # Si no hay dirección anterior, elegir aleatoriamente
            next_node = random.choice(neighbors)
        
        # Asegurar que no sea el mismo nodo
        if next_node == NodoActual:
            self.consecutive_repeats += 1
            if self.consecutive_repeats >= self.max_consecutive_repeats:
                # Forzar cambio a un nodo diferente
                if neighbors:
                    next_node = random.choice(neighbors)
                    self.consecutive_repeats = 0
        else:
            self.consecutive_repeats = 0
        
        # Registrar nodo visitado
        self.visited_nodes.add(next_node)
        
        # Ocasionalmente volver al nodo 0 para "reiniciar" la exploración
        # Pero solo si no es el mismo nodo actual y con menor probabilidad
        if random.random() < 0.02 and NodoActual != 0 and next_node != 0:  
            print(f"El agente {self.idx} regresó aleatoriamente al nodo 0")
            return 0
        
        # Asegurar que el nodo final no sea el mismo que el actual
        if next_node == NodoActual:
            if neighbors:
                next_node = random.choice(neighbors)
            
        return next_node

    def update(self, delta, other_lifters=None):
        """
        Actualiza el estado y posición del lifter.
        
        Args:
            delta: Delta de tiempo para animaciones
            other_lifters: Lista de otros lifters para detección de colisiones (opcional)
        """
        # Guardar estado y nodo anteriores para detectar cambios
        previous_state = self.status
        previous_node = self.currentNode
        
        if self.workCompleted:
            if random.random() < 0.01:
                print(f"El agente {self.idx} - Trabajo completado. Esperando en el nodo 0.")
            return
        
        if other_lifters is None:
            other_lifters = []
        
        # Asegurar que nextNode sea diferente de currentNode
        if self.nextNode == self.currentNode:
            print(f"El agente {self.idx}: nextNode igual a currentNode ({self.currentNode}), obteniendo nuevo nodo...")
            self.nextNode = self.RetrieveNextNodePath(self.currentNode)
            # Si aún es el mismo, forzar un vecino diferente
            if self.nextNode == self.currentNode:
                neighbors = []
                for i in range(len(A)):
                    if A[self.currentNode][i] == 1 and i != self.currentNode:
                        neighbors.append(i)
                if neighbors:
                    self.nextNode = random.choice(neighbors)
                    print(f"El agente {self.idx}: Forzando movimiento a nodo vecino {self.nextNode}")
                else:
                    print(f"El agente {self.idx}: No hay vecinos disponibles, permaneciendo en nodo {self.currentNode}")
        
        Direccion, Distancia = self.ComputeDirection(self.Position, self.nextNode)
        
        # Verificar si el siguiente nodo está fuera del área permitida
        if self.area_limit is not None:
            next_node_pos = NodosVisita[self.nextNode]
            if (next_node_pos[0] < self.area_limit['min'] or next_node_pos[0] > self.area_limit['max'] or
                next_node_pos[2] < self.area_limit['min'] or next_node_pos[2] > self.area_limit['max']):
                # El nodo está fuera del área, buscar un nodo alternativo dentro del área
                self._find_alternative_node_within_area()
                Direccion, Distancia = self.ComputeDirection(self.Position, self.nextNode)
        
        # Si la distancia es muy pequeña o cero, y estamos en el mismo nodo, forzar movimiento
        if Distancia < 0.1 and self.nextNode == self.currentNode:
            print(f"El agente {self.idx}: Distancia muy pequeña ({Distancia:.2f}) y mismo nodo, obteniendo nuevo nodo...")
            self.nextNode = self.RetrieveNextNodePath(self.currentNode)
            # Si aún es el mismo, forzar un vecino diferente
            if self.nextNode == self.currentNode:
                neighbors = []
                for i in range(len(A)):
                    if A[self.currentNode][i] == 1 and i != self.currentNode:
                        neighbors.append(i)
                if neighbors:
                    self.nextNode = random.choice(neighbors)
                    print(f"El agente {self.idx}: Forzando movimiento a nodo vecino {self.nextNode}")
            Direccion, Distancia = self.ComputeDirection(self.Position, self.nextNode)
        
        if Distancia < 2:
            # Para método aleatorio: guardar la dirección actual antes de cambiar de nodo
            if self.method == "random" and numpy.linalg.norm(Direccion) > 0.001:
                self.last_direction = Direccion.copy()
                self.last_direction[1] = 0  # Ignorar componente Y
                norm = numpy.linalg.norm(self.last_direction)
                if norm > 0:
                    self.last_direction /= norm
            
            print(f"El agente {self.idx} está calculando un nuevo nodo...")
            self.currentNode = self.nextNode
            self.movements += 1

            if self.returningToPickup:
                print(f"El agente {self.idx} regresó al nodo de recolección {self.lastPickupNode}.")
                self.returningToPickup = False
                self.lastPickupNode = None
                self.nextNode = self.RetrieveNextNodePath(self.currentNode)
            else:
                # Para método aleatorio: reducir frecuencia de cambios cuando está cerca del borde
                if self.method == "random" and self.area_limit is not None:
                    dist_to_x_min = self.Position[0] - self.area_limit['min']
                    dist_to_x_max = self.area_limit['max'] - self.Position[0]
                    dist_to_z_min = self.Position[2] - self.area_limit['min']
                    dist_to_z_max = self.area_limit['max'] - self.Position[2]
                    min_dist_to_edge = min(dist_to_x_min, dist_to_x_max, dist_to_z_min, dist_to_z_max)
                    
                    # Si está muy cerca del borde y el cooldown no ha terminado, mantener dirección similar
                    if min_dist_to_edge < 10 and self.node_change_cooldown > 0:
                        self.node_change_cooldown -= 1
                        # Buscar un vecino en dirección similar
                        neighbors = []
                        for i in range(len(A)):
                            if A[self.currentNode][i] == 1 and i != self.currentNode:
                                neighbors.append(i)
                        if neighbors and self.last_direction is not None:
                            # Elegir el vecino más cercano a la dirección actual
                            best_neighbor = None
                            best_similarity = -1
                            for n in neighbors:
                                neighbor_pos = NodosVisita[n]
                                dir_to_n = neighbor_pos - self.Position
                                dir_to_n[1] = 0
                                norm = numpy.linalg.norm(dir_to_n)
                                if norm > 0:
                                    dir_to_n /= norm
                                    similarity = numpy.dot(self.last_direction, dir_to_n)
                                    if similarity > best_similarity:
                                        best_similarity = similarity
                                        best_neighbor = n
                            if best_neighbor is not None and best_similarity > 0.3:
                                self.nextNode = best_neighbor
                            else:
                                self.nextNode = self.RetrieveNextNodePath(self.currentNode)
                        else:
                            self.nextNode = self.RetrieveNextNodePath(self.currentNode)
                    else:
                        # Resetear cooldown después de cambiar de nodo
                        self.node_change_cooldown = 3  # Esperar 3 frames antes de cambiar de nodo otra vez
                        self.nextNode = self.RetrieveNextNodePath(self.currentNode)
                else:
                    self.nextNode = self.RetrieveNextNodePath(self.currentNode)
            
            # Asegurar que el siguiente nodo no sea el mismo que el actual
            if self.nextNode == self.currentNode:
                neighbors = []
                for i in range(len(A)):
                    if A[self.currentNode][i] == 1 and i != self.currentNode:
                        neighbors.append(i)
                if neighbors:
                    self.nextNode = random.choice(neighbors)
                    print(f"El agente {self.idx}: Evitando nodo repetido, moviendo a nodo {self.nextNode}")
                else:
                    print(f"El agente {self.idx}: Advertencia: No hay vecinos disponibles en nodo {self.currentNode}")

        mssg = "Agent:%d \t Method:%s \t State:%s \t Position:[%0.2f,0,%0.2f] \t NodoActual:%d \t NodoSiguiente:%d" % (
            self.idx, self.method, self.status, self.Position[0], self.Position[-1], self.currentNode, self.nextNode
        )
        if self.method == "planned":
            mssg += f" \t Route:{self.route}"
        print(mssg)

        match self.status:
            case "searching":
                # Verificar que la dirección no sea cero
                if numpy.linalg.norm(Direccion) < 0.001:
                    print(f"El agente {self.idx}: Dirección cero detectada, forzando nuevo nodo...")
                    self.nextNode = self.RetrieveNextNodePath(self.currentNode)
                    if self.nextNode == self.currentNode:
                        neighbors = []
                        for i in range(len(A)):
                            if A[self.currentNode][i] == 1 and i != self.currentNode:
                                neighbors.append(i)
                        if neighbors:
                            self.nextNode = random.choice(neighbors)
                    Direccion, Distancia = self.ComputeDirection(self.Position, self.nextNode)
                
                # Aplicar evitación de colisiones antes de mover
                safe_direction = self.applyCollisionAvoidance(Direccion, other_lifters)
                
                # Verificar que safe_direction no sea cero
                if numpy.linalg.norm(safe_direction) < 0.001:
                    print(f"El agente {self.idx}: Dirección segura cero, usando dirección original...")
                    safe_direction = Direccion
                    if numpy.linalg.norm(safe_direction) < 0.001:
                        # Si aún es cero, usar dirección aleatoria
                        safe_direction = numpy.random.rand(3)
                        safe_direction[1] = 0
                        norm = numpy.linalg.norm(safe_direction)
                        if norm > 0:
                            safe_direction /= norm
                
                new_position = self.Position + safe_direction * self.vel
                
                # Limitar movimiento al área de simulación si está definida
                # Si se alcanza el borde, rebotar hacia el interior (no seguir el borde)
                if self.area_limit is not None:
                    hit_x_boundary = False
                    hit_z_boundary = False
                    boundary_threshold = 5.0  # Distancia mínima del borde para considerar que está cerca
                    
                    # Verificación preventiva: si está muy cerca del borde, cambiar de dirección antes de tocarlo
                    # Solo para método planeado, para aleatorio se maneja diferente
                    if self.method == "planned":
                        dist_to_x_min = self.Position[0] - self.area_limit['min']
                        dist_to_x_max = self.area_limit['max'] - self.Position[0]
                        dist_to_z_min = self.Position[2] - self.area_limit['min']
                        dist_to_z_max = self.area_limit['max'] - self.Position[2]
                        
                        min_dist_to_edge = min(dist_to_x_min, dist_to_x_max, dist_to_z_min, dist_to_z_max)
                        
                        # Si está muy cerca del borde y se está moviendo hacia él, cambiar de dirección
                        if min_dist_to_edge < boundary_threshold:
                            # Verificar si la dirección actual lo llevaría más cerca del borde
                            will_hit_edge = False
                            if dist_to_x_min < boundary_threshold and safe_direction[0] < 0:
                                will_hit_edge = True
                            elif dist_to_x_max < boundary_threshold and safe_direction[0] > 0:
                                will_hit_edge = True
                            elif dist_to_z_min < boundary_threshold and safe_direction[2] < 0:
                                will_hit_edge = True
                            elif dist_to_z_max < boundary_threshold and safe_direction[2] > 0:
                                will_hit_edge = True
                            
                            if will_hit_edge:
                                print(f"El agente {self.idx}: Cerca del borde ({min_dist_to_edge:.2f}), buscando nodo hacia el interior...")
                                # Buscar un nodo hacia el interior antes de tocar el borde
                                valid_nodes = []
                                for i, node_pos in enumerate(NodosVisita):
                                    if (self.area_limit['min'] <= node_pos[0] <= self.area_limit['max'] and
                                        self.area_limit['min'] <= node_pos[2] <= self.area_limit['max']):
                                        # Calcular si el nodo está hacia el interior
                                        node_dist_to_x_min = node_pos[0] - self.area_limit['min']
                                        node_dist_to_x_max = self.area_limit['max'] - node_pos[0]
                                        node_dist_to_z_min = node_pos[2] - self.area_limit['min']
                                        node_dist_to_z_max = self.area_limit['max'] - node_pos[2]
                                        node_min_dist = min(node_dist_to_x_min, node_dist_to_x_max, node_dist_to_z_min, node_dist_to_z_max)
                                        
                                        # Preferir nodos que estén más hacia el interior que la posición actual
                                        if node_min_dist > min_dist_to_edge:
                                            valid_nodes.append((i, node_min_dist))
                                
                                if valid_nodes:
                                    valid_nodes.sort(key=lambda x: x[1], reverse=True)
                                    candidates = valid_nodes[:min(3, len(valid_nodes))]
                                    chosen_node = random.choice(candidates)[0]
                                    self.nextNode = chosen_node
                                    Direccion, Distancia = self.ComputeDirection(self.Position, self.nextNode)
                                    safe_direction = self.applyCollisionAvoidance(Direccion, other_lifters)
                                    new_position = self.Position + safe_direction * self.vel
                    
                    # Verificar límites en X
                    if new_position[0] < self.area_limit['min']:
                        new_position[0] = self.area_limit['min']
                        hit_x_boundary = True
                    elif new_position[0] > self.area_limit['max']:
                        new_position[0] = self.area_limit['max']
                        hit_x_boundary = True
                    
                    # Verificar límites en Z
                    if new_position[2] < self.area_limit['min']:
                        new_position[2] = self.area_limit['min']
                        hit_z_boundary = True
                    elif new_position[2] > self.area_limit['max']:
                        new_position[2] = self.area_limit['max']
                        hit_z_boundary = True
                    
                    # Si toca cualquier borde, buscar un nodo hacia el interior
                    if hit_x_boundary or hit_z_boundary:
                        print(f"El agente {self.idx}: Tocó el borde (X:{hit_x_boundary}, Z:{hit_z_boundary}), buscando nodo hacia el interior...")
                        
                        # Buscar nodos válidos dentro del área que estén hacia el interior
                        valid_nodes = []
                        for i, node_pos in enumerate(NodosVisita):
                            # Verificar que el nodo esté dentro del área
                            if (self.area_limit['min'] <= node_pos[0] <= self.area_limit['max'] and
                                self.area_limit['min'] <= node_pos[2] <= self.area_limit['max']):
                                # Calcular distancia desde el borde hacia el interior
                                dist_from_x_min = node_pos[0] - self.area_limit['min']
                                dist_from_x_max = self.area_limit['max'] - node_pos[0]
                                dist_from_z_min = node_pos[2] - self.area_limit['min']
                                dist_from_z_max = self.area_limit['max'] - node_pos[2]
                                
                                # Preferir nodos que estén más hacia el interior (lejos de los bordes)
                                min_dist_from_edge = min(dist_from_x_min, dist_from_x_max, dist_from_z_min, dist_from_z_max)
                                
                                # Si tocó ambos bordes (esquina), preferir nodos hacia el centro
                                if hit_x_boundary and hit_z_boundary:
                                    # Calcular distancia al centro
                                    dist_to_center = math.sqrt(node_pos[0]**2 + node_pos[2]**2)
                                    # Preferir nodos que estén hacia el interior en ambas direcciones
                                    if ((self.Position[0] <= self.area_limit['min'] and node_pos[0] > self.Position[0]) or \
                                        (self.Position[0] >= self.area_limit['max'] and node_pos[0] < self.Position[0])) and \
                                       ((self.Position[2] <= self.area_limit['min'] and node_pos[2] > self.Position[2]) or \
                                        (self.Position[2] >= self.area_limit['max'] and node_pos[2] < self.Position[2])):
                                        valid_nodes.append((i, min_dist_from_edge + dist_to_center * 0.1))
                                # Si tocó borde X, preferir nodos que estén más hacia el interior en X
                                elif hit_x_boundary:
                                    if (self.Position[0] <= self.area_limit['min'] and node_pos[0] > self.Position[0]) or \
                                       (self.Position[0] >= self.area_limit['max'] and node_pos[0] < self.Position[0]):
                                        valid_nodes.append((i, min_dist_from_edge))
                                # Si tocó borde Z, preferir nodos que estén más hacia el interior en Z
                                elif hit_z_boundary:
                                    if (self.Position[2] <= self.area_limit['min'] and node_pos[2] > self.Position[2]) or \
                                       (self.Position[2] >= self.area_limit['max'] and node_pos[2] < self.Position[2]):
                                        valid_nodes.append((i, min_dist_from_edge))
                        
                        if valid_nodes:
                            # Ordenar por distancia desde el borde (preferir los más hacia el interior)
                            valid_nodes.sort(key=lambda x: x[1], reverse=True)
                            # Tomar uno de los mejores candidatos (top 3)
                            candidates = valid_nodes[:min(3, len(valid_nodes))]
                            chosen_node = random.choice(candidates)[0]
                            self.nextNode = chosen_node
                            print(f"El agente {self.idx}: Cambiando a nodo {self.nextNode} hacia el interior")
                            # Recalcular dirección hacia el nuevo nodo
                            Direccion, Distancia = self.ComputeDirection(self.Position, self.nextNode)
                            safe_direction = self.applyCollisionAvoidance(Direccion, other_lifters)
                        else:
                            # Si no hay nodos válidos, rebotar hacia el centro
                            print(f"El agente {self.idx}: No hay nodos válidos, rebotando hacia el centro")
                            center_dir = numpy.array([-self.Position[0], 0, -self.Position[2]])
                            magnitude = numpy.linalg.norm(center_dir)
                            if magnitude > 0:
                                safe_direction = center_dir / magnitude
                            else:
                                # Si ya está en el centro, dirección aleatoria hacia el interior
                                safe_direction = numpy.random.rand(3)
                                safe_direction[1] = 0
                                # Asegurar que apunte hacia el interior
                                if self.Position[0] < 0:
                                    safe_direction[0] = abs(safe_direction[0])
                                else:
                                    safe_direction[0] = -abs(safe_direction[0])
                                if self.Position[2] < 0:
                                    safe_direction[2] = abs(safe_direction[2])
                                else:
                                    safe_direction[2] = -abs(safe_direction[2])
                                norm = numpy.linalg.norm(safe_direction)
                                if norm > 0:
                                    safe_direction /= norm
                    
                    # Renormalizar la dirección después de los cambios
                    magnitude = numpy.linalg.norm(safe_direction)
                    if magnitude > 0:
                        safe_direction /= magnitude
                    else:
                        # Fallback: dirección aleatoria
                        safe_direction = numpy.random.rand(3)
                        safe_direction[1] = 0
                        safe_direction /= numpy.linalg.norm(safe_direction)
                
                self.Position = new_position
                self.Direction = safe_direction
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
                    # Aplicar evitación de colisiones también al entregar
                    safe_direction = self.applyCollisionAvoidance(self.Direction, other_lifters)
                    newX = self.Position[0] + safe_direction[0] * self.vel
                    newZ = self.Position[2] + safe_direction[2] * self.vel
                    
                    # Limitar movimiento al área de simulación
                    if self.area_limit is not None:
                        hit_x_boundary = False
                        hit_z_boundary = False
                        
                        # Verificar límites en X
                        if newX < self.area_limit['min']:
                            newX = self.area_limit['min']
                            hit_x_boundary = True
                        elif newX > self.area_limit['max']:
                            newX = self.area_limit['max']
                            hit_x_boundary = True
                        
                        # Verificar límites en Z
                        if newZ < self.area_limit['min']:
                            newZ = self.area_limit['min']
                            hit_z_boundary = True
                        elif newZ > self.area_limit['max']:
                            newZ = self.area_limit['max']
                            hit_z_boundary = True
                        
                        # Si toca un borde, redirigir hacia el centro (incinerador)
                        if hit_x_boundary or hit_z_boundary:
                            # Calcular dirección hacia el centro
                            center_dir = numpy.array([-self.Position[0], 0, -self.Position[2]])
                            magnitude = numpy.linalg.norm(center_dir)
                            if magnitude > 0:
                                safe_direction = center_dir / magnitude
                                # Aplicar evitación de colisiones a la nueva dirección
                                safe_direction = self.applyCollisionAvoidance(safe_direction, other_lifters)
                            else:
                                # Si ya está en el centro, usar dirección original
                                pass
                        
                        # Renormalizar la dirección
                        magnitude = numpy.linalg.norm(safe_direction)
                        if magnitude > 0:
                            safe_direction /= magnitude
                        else:
                            # Fallback: dirección hacia el centro
                            center_dir = numpy.array([-self.Position[0], 0, -self.Position[2]])
                            magnitude = numpy.linalg.norm(center_dir)
                            if magnitude > 0:
                                safe_direction = center_dir / magnitude
                    else:
                        # Fallback al comportamiento original si no hay límite
                        if newX - 10 < -self.dim or newX + 10 > self.dim:
                            safe_direction[0] *= -1
                        if newZ - 10 < -self.dim or newZ + 10 > self.dim:
                            safe_direction[2] *= -1
                    
                    self.Position[0] = newX
                    self.Position[2] = newZ
                    self.Direction = safe_direction
                    self.angle = math.acos(self.Direction[0]) * 180 / math.pi
                    if self.Direction[2] > 0:
                        self.angle = 360 - self.angle
            case "dropping":
                if self.platformHeight <= -1.5:
                    if self.lastPickupNode is not None:
                        if self.isFinalNode(self.lastPickupNode):
                            self.finalNodeWithTrash = True
                            print(f"El agente {self.idx} encontró basura en el nodo final {self.lastPickupNode}")
                        
                        self.nextNode = self.lastPickupNode
                        self.returningToPickup = True
                        print(f"El agente {self.idx} regresó al nodo {self.lastPickupNode} para buscar más basura")
                    self.status = "searching"
                else:
                    self.platformHeight -= delta

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