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

    # Procesar métodos individuales si se especificaron
    methods_list = []
    if Options.methods:
        # Parsear métodos individuales separados por comas
        methods_list = [m.strip().lower() for m in Options.methods.split(',')]
        # Validar que todos sean válidos
        for m in methods_list:
            if m not in ["planned", "random"]:
                raise ValueError(f"Método inválido: {m}. Debe ser 'planned' o 'random'")
        # Validar que haya tantos métodos como lifters
        if len(methods_list) != Options.lifters:
            raise ValueError(f"Debe haber {Options.lifters} métodos (uno por cada lifter), pero se proporcionaron {len(methods_list)}")
    else:
        # Usar el mismo método para todos
        methods_list = [Options.method] * Options.lifters

    # Determinar tamaño de la matriz para los nodos de los lifters
    # Para planned siempre usa 5x5, para random usa Options.M
    has_random = any(m == "random" for m in methods_list) if Options.methods else (Options.method == "random")
    M_lifters = Options.M if has_random else 5
    
    # M para basuras: siempre usa Options.M para definir el área donde aparecen
    M_basuras = Options.M

    Lifter.init_world(M_lifters)
    total_nodes = M_lifters * M_lifters

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

    # Generar posiciones de basura dentro del área definida por M_basuras
    # El área de basuras se calcula basándose en M_basuras
    # Si M_basuras es pequeño, las basuras aparecen en un área pequeña
    # Si M_basuras es grande, las basuras aparecen en un área grande
    DimBoard = Settings.DimBoard
    # Calcular el área donde pueden aparecer las basuras basado en M_basuras
    # Similar a como se calculan los nodos, pero para el área de basuras
    if M_basuras > 1:
        basura_spacing = (2 * DimBoard) / (M_basuras - 1)
        basura_area_size = basura_spacing * (M_basuras - 1)
    else:
        basura_area_size = 2 * DimBoard
    
    # El área de basuras va desde -basura_area_size/2 hasta +basura_area_size/2
    # Centrado en el origen
    basura_min = -basura_area_size / 2
    basura_max = basura_area_size / 2
    
    NodosCarga = []
    for i in range(Options.Basuras):
        # Generar posición aleatoria dentro del área definida por M_basuras
        x = random.uniform(basura_min, basura_max)
        z = random.uniform(basura_min, basura_max)
        NodosCarga.append([x, 0, z])
    
    print(f"Área de basuras: de {basura_min:.2f} a {basura_max:.2f} (basado en M={M_basuras})")

    # Crear lifters con sus métodos individuales
    # Inicializar lifters en el CENTRO (0, 0, 0)
    # Calcular qué nodo está más cerca del centro
    center_pos = numpy.array([0.0, 0.0, 0.0])
    CurrentNode = 0
    if len(Lifter.NodosVisita) > 0:
        # Encontrar el nodo más cercano al centro
        distances = numpy.linalg.norm(Lifter.NodosVisita - center_pos, axis=1)
        CurrentNode = numpy.argmin(distances)
        center_pos = Lifter.NodosVisita[CurrentNode].copy()
        print(f"Nodo central encontrado: {CurrentNode} en posición {center_pos}")
    
    Positions = numpy.zeros((Options.lifters, 3))
    for i, p in enumerate(Positions):
        # Todos empiezan en el centro (0,0,0) o el nodo más cercano al centro
        Positions[i] = center_pos.copy()
        method_for_lifter = methods_list[i]
        # Aumentar velocidad de los lifters (de 0.7 a 1.5 para hacerlo más rápido)
        lifters.append(Lifter.Lifter(Settings.DimBoard, 1.5, textures, i, Positions[i], CurrentNode, method_for_lifter))
        print(f"Lifter {i} inicializado con método: {method_for_lifter} en posición {Positions[i]} (nodo {CurrentNode})")

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
    for c in lifters:
        for b in basuras:
            distance = math.sqrt((b.Position[0] - c.Position[0])**2 + (b.Position[2] - c.Position[2])**2)
            if distance <= c.radiusCol:
                if c.status == "searching" and b.alive:
                    b.alive = False
                    c.status = "lifting"
                    c.lastPickupNode = c.currentNode

def display():
    global lifters, basuras, delta, work_completed_time, all_work_done, final_trash_count, Options_global

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Actualizar lifters pasando la lista de otros lifters para evitar colisiones
    for i, obj in enumerate(lifters):
        # Crear lista de otros lifters (excluyendo el actual)
        other_lifters = [lifter for j, lifter in enumerate(lifters) if j != i]
        obj.update(delta, other_lifters)
        obj.draw()

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
        # Contar métodos usados
        planned_count = sum(1 for lifter in lifters if lifter.method == "planned")
        random_count = sum(1 for lifter in lifters if lifter.method == "random")
        
        print("\n" + "="*60)
        print("RESUMEN DE LA SIMULACIÓN")
        print("="*60)
        if Options.methods:
            print(f"{'Métodos usados:':<25} Mixto (--methods especificado)")
            print(f"{'  - Planeados:':<25} {planned_count}")
            print(f"{'  - Aleatorios:':<25} {random_count}")
        else:
            print(f"{'Tipo de experimento:':<25} {Options.method}")
        print(f"{'Tmax (límite):':<25} {Options.Tmax:.1f} s")
        M_used = Lifter.matrix_size if Lifter.matrix_size else (5 if Options.method == "planned" else Options.M)
        print(f"{'M (tamaño matriz):':<25} {M_used}")
        print(f"{'N (número de agentes):':<25} {Options.lifters}")
        print(f"{'Celdas inicialmente sucias:':<25} {initial_dirty_pct:.1f}% ({initial_trash_count}/{total_nodes})")
        print(f"{'Celdas limpias al final:':<25} {final_clean_pct:.1f}% ({total_nodes - final_trash_count}/{total_nodes})")
        print(f"{'Tiempo real de simulación:':<25} {real_duration:.2f} s")
        print(f"{'Movimientos totales:':<25} {total_movements}")
        print("="*60)