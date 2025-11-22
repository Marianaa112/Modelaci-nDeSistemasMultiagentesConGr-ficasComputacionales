import yaml, pygame, random, glob, math, numpy, time
import Lifter 
from Basura import Basura
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# === Variables globales para métricas del resumen ===
textures = []
lifters = []
basuras = []
delta = 0
start_time = 0
simulation_start_time = None
work_completed_time = None
initial_trash_count = 0
final_trash_count = 0
all_work_done = False
Options_global = None  

def GeneracionDeNodos():
    print("Generación de nodos no implementada aún.")

def loadSettingsYAML(File):
    class Settings: pass
    with open(File) as f:
        docs = yaml.load_all(f, Loader=yaml.FullLoader)
        for doc in docs:
            for k, v in doc.items():
                setattr(Settings, k, v)
    return Settings

Settings = loadSettingsYAML("Settings.yaml")

def Axis():
    glShadeModel(GL_FLAT)
    glLineWidth(3.0)
    glColor3f(1.0,0.0,0.0)
    glBegin(GL_LINES)
    glVertex3f(X_MIN,0.0,0.0)
    glVertex3f(X_MAX,0.0,0.0)
    glEnd()
    glColor3f(0.0,1.0,0.0)
    glBegin(GL_LINES)
    glVertex3f(0.0,Y_MIN,0.0)
    glVertex3f(0.0,Y_MAX,0.0)
    glEnd()
    glColor3f(0.0,0.0,1.0)
    glBegin(GL_LINES)
    glVertex3f(0.0,0.0,Z_MIN)
    glVertex3f(0.0,0.0,Z_MAX)
    glEnd()
    glLineWidth(1.0)

def Texturas(filepath):
    global textures
    textures.append(glGenTextures(1))
    id = len(textures) - 1
    glBindTexture(GL_TEXTURE_2D, textures[id])
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    image = pygame.image.load(filepath).convert()
    w, h = image.get_rect().size
    image_data = pygame.image.tostring(image, "RGBA")
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
    glGenerateMipmap(GL_TEXTURE_2D)

def Init(Options):
    global textures, basuras, lifters, start_time, initial_trash_count

    # Determinar tamaño de la matriz
    if Options.method == "random":
        M = Options.M
    else:
        M = 5  # Hardcoded para planned

    Lifter.init_world(M)
    total_nodes = M * M

    screen = pygame.display.set_mode((Settings.screen_width, Settings.screen_height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("OpenGL: cubos")
    start_time = time.time()

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(Settings.FOVY, Settings.screen_width / Settings.screen_height, Settings.ZNEAR, Settings.ZFAR)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(
        Settings.EYE_X,
        Settings.EYE_Y,
        Settings.EYE_Z,
        Settings.CENTER_X,
        Settings.CENTER_Y,
        Settings.CENTER_Z,
        Settings.UP_X,
        Settings.UP_Y,
        Settings.UP_Z)
    glClearColor(0, 0, 0, 0)
    glEnable(GL_DEPTH_TEST)
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    for File in glob.glob(Settings.Materials + "*.*"):
        Texturas(File)

    # Generar posiciones de basura
    NodosCarga = []
    for i in range(Options.Basuras):
        if total_nodes <= 1:
            nodo_aleatorio = 0
        else:
            nodo_aleatorio = random.randint(1, total_nodes - 1)
        x = Lifter.NodosVisita[nodo_aleatorio][0]
        z = Lifter.NodosVisita[nodo_aleatorio][2]
        NodosCarga.append([x, 0, z])

    # Crear lifters
    # Para método random, distribuir los lifters en diferentes nodos iniciales
    if Options.method == "random":
        # Distribuir los lifters en diferentes nodos para evitar que todos empiecen en el mismo punto
        total_nodes = M * M
        initial_nodes = []
        for i in range(Options.lifters):
            # Distribuir de manera más uniforme
            if i == 0:
                initial_nodes.append(0)  # El primero siempre en el nodo 0
            else:
                # Los demás en nodos distribuidos
                node_offset = (i * (total_nodes - 1)) // Options.lifters
                initial_nodes.append(min(node_offset, total_nodes - 1))
    else:
        # Para planned, todos empiezan en el nodo 0
        initial_nodes = [0] * Options.lifters
    
    Positions = numpy.zeros((Options.lifters, 3))
    for i, p in enumerate(Positions):
        currentNode = initial_nodes[i]
        # Establecer la posición inicial en el nodo correspondiente
        if currentNode < len(Lifter.NodosVisita):
            p[0] = Lifter.NodosVisita[currentNode][0]
            p[1] = Lifter.NodosVisita[currentNode][1]
            p[2] = Lifter.NodosVisita[currentNode][2]
            # Si es random y hay múltiples lifters en el mismo nodo, agregar pequeño offset
            if Options.method == "random":
                # Contar cuántos lifters anteriores están en el mismo nodo
                same_node_count = sum(1 for j in range(i) if initial_nodes[j] == currentNode)
                if same_node_count > 0:
                    # Pequeño offset para evitar superposición exacta
                    angle_offset = (same_node_count * 2 * math.pi) / 8  # Distribuir en círculo
                    p[0] += math.cos(angle_offset) * 1.5
                    p[2] += math.sin(angle_offset) * 1.5
        lifters.append(Lifter.Lifter(Settings.DimBoard, 0.7, textures, i, p, currentNode, Options.method))

    # Crear basuras
    for i, n in enumerate(NodosCarga):
        basuras.append(Basura(Settings.DimBoard, 1, textures, 3, i, n))

    initial_trash_count = len(basuras)

def planoText():
    glColor(1.0, 1.0, 1.0)
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0)
    glVertex3d(-Settings.DimBoard, 0, -Settings.DimBoard)
    glTexCoord2f(0.0, 1.0)
    glVertex3d(-Settings.DimBoard, 0, Settings.DimBoard)
    glTexCoord2f(1.0, 1.0)
    glVertex3d(Settings.DimBoard, 0, Settings.DimBoard)
    glTexCoord2f(1.0, 0.0)
    glVertex3d(Settings.DimBoard, 0, -Settings.DimBoard)
    glEnd()

def checkCollisions():
    # Detectar colisiones entre lifters y basura
    for c in lifters:
        for b in basuras:
            distance = math.sqrt((b.Position[0] - c.Position[0])**2 + (b.Position[2] - c.Position[2])**2)
            if distance <= c.radiusCol:
                if c.status == "searching" and b.alive:
                    b.alive = False
                    c.status = "lifting"
                    c.lastPickupNode = c.currentNode
    
    # Detectar colisiones entre lifters (solo para random, excepto en área de tirar basura)
    for i, lifter1 in enumerate(lifters):
        for j, lifter2 in enumerate(lifters):
            if i >= j:  # Evitar comparar dos veces el mismo par
                continue
            
            distance = math.sqrt((lifter2.Position[0] - lifter1.Position[0])**2 + 
                               (lifter2.Position[2] - lifter1.Position[2])**2)
            
            # Radio de detección para evitar colisiones (ajustado para balance entre distancia y fluidez)
            detection_radius = lifter1.radiusCol * 2.5  # Radio moderado para mantener distancia sin ser excesivo
            
            # Solo evitar colisiones si al menos uno es random y están fuera del área de tirar basura
            if distance <= detection_radius:
                in_trash_area1 = lifter1.isInTrashArea()
                in_trash_area2 = lifter2.isInTrashArea()
                
                # Si ambos están en el área de tirar basura, permitir colisiones/amontonamiento
                if in_trash_area1 and in_trash_area2:
                    continue
                
                # Si ambos son random y al menos uno está fuera del área de tirar basura
                if lifter1.method == "random" and lifter2.method == "random":
                    if not in_trash_area1 and not in_trash_area2:
                        # Ambos están fuera, ambos deben evitar
                        lifter1.nearby_lifter_detected = True
                        lifter2.nearby_lifter_detected = True
                    elif not in_trash_area1:
                        # Solo lifter1 está fuera, solo él evita
                        lifter1.nearby_lifter_detected = True
                    elif not in_trash_area2:
                        # Solo lifter2 está fuera, solo él evita
                        lifter2.nearby_lifter_detected = True
                elif lifter1.method == "random" and not in_trash_area1:
                    # Lifter1 es random y está fuera, debe evitar
                    lifter1.nearby_lifter_detected = True
                elif lifter2.method == "random" and not in_trash_area2:
                    # Lifter2 es random y está fuera, debe evitar
                    lifter2.nearby_lifter_detected = True

def display():
    global lifters, basuras, delta, work_completed_time, all_work_done, final_trash_count, Options_global

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    for obj in lifters:
        obj.draw()
        obj.update(delta)

    # Contar basura viva
    alive_trash = sum(1 for b in basuras if b.alive)
    final_trash_count = alive_trash

    # Verificar si el trabajo está completo
    if alive_trash == 0 and not all_work_done:
        if Options_global.method == "planned":
            all_finished = all(lifter.workCompleted for lifter in lifters)
        else:  # random: si no hay basura, terminó
            all_finished = True

        if all_finished:
            all_work_done = True
            work_completed_time = time.time() - simulation_start_time
            print("\nToda la basura ha sido recolectada")

    # Dibujar incinerador
    glColor3f(1.0, 0.4, 0.7)
    square_size = 20.0
    half_size = square_size / 2.0
    glBegin(GL_QUADS)
    glVertex3d(-half_size, 0.5, -half_size)
    glVertex3d(-half_size, 0.5, half_size)
    glVertex3d(half_size, 0.5, half_size)
    glVertex3d(half_size, 0.5, -half_size)
    glEnd()

    # Dibujar basuras
    for obj in basuras:
        obj.draw()

    # Dibujar plano
    planoText()
    glColor3f(0.3, 0.3, 0.3)
    glBegin(GL_QUADS)
    glVertex3d(-Settings.DimBoard, 0, -Settings.DimBoard)
    glVertex3d(-Settings.DimBoard, 0, Settings.DimBoard)
    glVertex3d(Settings.DimBoard, 0, Settings.DimBoard)
    glVertex3d(Settings.DimBoard, 0, -Settings.DimBoard)
    glEnd()

    # Dibujar paredes
    wall_height = 50.0
    glColor3f(0.8, 0.8, 0.8)
    # Left
    glBegin(GL_QUADS)
    glVertex3d(-Settings.DimBoard, 0, -Settings.DimBoard)
    glVertex3d(-Settings.DimBoard, 0, Settings.DimBoard)
    glVertex3d(-Settings.DimBoard, wall_height, Settings.DimBoard)
    glVertex3d(-Settings.DimBoard, wall_height, -Settings.DimBoard)
    glEnd()
    # Right
    glBegin(GL_QUADS)
    glVertex3d(Settings.DimBoard, 0, -Settings.DimBoard)
    glVertex3d(Settings.DimBoard, 0, Settings.DimBoard)
    glVertex3d(Settings.DimBoard, wall_height, Settings.DimBoard)
    glVertex3d(Settings.DimBoard, wall_height, -Settings.DimBoard)
    glEnd()
    # Front
    glBegin(GL_QUADS)
    glVertex3d(-Settings.DimBoard, 0, Settings.DimBoard)
    glVertex3d(Settings.DimBoard, 0, Settings.DimBoard)
    glVertex3d(Settings.DimBoard, wall_height, Settings.DimBoard)
    glVertex3d(-Settings.DimBoard, wall_height, Settings.DimBoard)
    glEnd()
    # Back
    glBegin(GL_QUADS)
    glVertex3d(-Settings.DimBoard, 0, -Settings.DimBoard)
    glVertex3d(Settings.DimBoard, 0, -Settings.DimBoard)
    glVertex3d(Settings.DimBoard, wall_height, -Settings.DimBoard)
    glVertex3d(-Settings.DimBoard, wall_height, -Settings.DimBoard)
    glEnd()

    checkCollisions()

def lookAt(theta):
    glLoadIdentity()
    rad = theta * math.pi / 180
    newX = Settings.EYE_X * math.cos(rad) + Settings.EYE_Z * math.sin(rad)
    newZ = -Settings.EYE_X * math.sin(rad) + Settings.EYE_Z * math.cos(rad)
    gluLookAt(
        newX,
        Settings.EYE_Y,
        newZ,
        Settings.CENTER_X,
        Settings.CENTER_Y,
        Settings.CENTER_Z,
        Settings.UP_X,
        Settings.UP_Y,
        Settings.UP_Z)

def Simulacion(Options):
    global delta, simulation_start_time, Options_global
    Options_global = Options
    simulation_start_time = time.time()

    theta = Options.theta
    radius = Options.radious
    delta = Options.Delta
    Init(Options)

    reloj = pygame.time.Clock()
    t_act = 0

    while True:
        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.type == pygame.QUIT:
                    pygame.quit()
                    return

        if Options.Tmax < (t_act / 30):
            print("T max reached*")
            break

        if keys[pygame.K_RIGHT]:
            theta = (theta + 1.0) % 360.0
            lookAt(theta)
        if keys[pygame.K_LEFT]:
            theta = (theta - 1.0) % 360.0
            lookAt(theta)

        display()
        pygame.display.flip()
        reloj.tick(30)
        t_act += 1

    pygame.quit()

    # === Calcular y mostrar resumen ===
    total_movements = sum(lifter.movements for lifter in lifters)
    total_nodes = Lifter.matrix_size ** 2 if Lifter.matrix_size else 25
    initial_dirty_pct = (initial_trash_count / total_nodes) * 100 if total_nodes > 0 else 0
    final_clean_pct = ((total_nodes - final_trash_count) / total_nodes) * 100 if total_nodes > 0 else 0

    real_duration = work_completed_time if all_work_done else (time.time() - simulation_start_time)

    if Options.resumen == "s":
        print("\n" + "="*60)
        print("RESUMEN DE LA SIMULACIÓN")
        print("="*60)
        print(f"{'Tipo de experimento:':<25} {Options.method}")
        print(f"{'Tmax (límite):':<25} {Options.Tmax:.1f} s")
        M_used = 5 if Options.method == "planned" else Options.M
        print(f"{'M (tamaño matriz):':<25} {M_used}")
        print(f"{'N (número de agentes):':<25} {Options.lifters}")
        print(f"{'Celdas inicialmente sucias:':<25} {initial_dirty_pct:.1f}% ({initial_trash_count}/{total_nodes})")
        print(f"{'Celdas limpias al final:':<25} {final_clean_pct:.1f}% ({total_nodes - final_trash_count}/{total_nodes})")
        print(f"{'Tiempo real de simulación:':<25} {real_duration:.2f} s")
        print(f"{'Movimientos totales:':<25} {total_movements}")
        print("="*60)