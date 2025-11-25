# Reto

Este es el codigo para realizar la base del Reto

## Instrucciones

### Ejecución básica

Para ejecutar el código puedes usar argumentos de línea de comandos o un archivo JSON:

**Con argumentos de línea de comandos:**

```bash
python Main.py Simulacion --lifters 3 --Basuras 6 --method random --Tmax 60 --M 5
```

**Con archivo JSON:**

```bash
python Main.py Simulacion --config config_example.json
```

**Combinando ambos (JSON + argumentos que sobrescriben):**

```bash
python Main.py Simulacion --config config_example.json --lifters 5 --Tmax 120
```

### Ejemplos de uso

**Simulación con método aleatorio (3 agentes, 6 basuras):**

```bash
python Main.py Simulacion --lifters 3 --Basuras 6 --method random --Tmax 60 --M 5
```

**Simulación con método planeado (2 agentes, 4 basuras):**

```bash
python Main.py Simulacion --lifters 2 --Basuras 4 --method planned --Tmax 60
```

**Simulación más rápida (aumentar Delta):**

```bash
python Main.py Simulacion --lifters 3 --Basuras 6 --method random --Delta 0.1 --Tmax 60 --M 5
```

**Usando archivo JSON:**

```bash
python Main.py Simulacion --config config_example.json
```

### Formato del archivo JSON

Puedes crear un archivo JSON con la configuración. Ejemplo (`config_example.json`):

```json
{
  "command": "Simulacion",
  "lifters": 3,
  "basuras": 6,
  "method": "random",
  "Tmax": 60,
  "M": 5,
  "Delta": 0.05,
  "theta": 0,
  "radious": 30,
  "resumen": "s"
}
```

**Nota:** Los argumentos de línea de comandos tienen prioridad sobre los valores del JSON. Esto te permite usar el JSON como base y sobrescribir valores específicos cuando sea necesario.

## Parámetros configurables:

```
--config, --json: archivo JSON con la configuración (opcional)
--lifters: número de agentes/lifters (obligatorio si no está en JSON)
--Basuras: número de objetos de basura (obligatorio si no está en JSON)
--Delta: velocidad de simulación (opcional, predeterminado: 0.05)
--theta: ángulo de rotación de cámara (opcional, predeterminado: 0)
--radious: radio de la cámara (opcional, predeterminado: 30)
--method: método de navegación - "planned" o "random" (predeterminado: planned)
--Tmax: duración máxima de la simulación en segundos (predeterminado: 60)
--M: tamaño de la matriz M×M para el método aleatorio (predeterminado: 5)
--resumen: mostrar resumen al final - "s" o "n" (predeterminado: "s")
```

## Funcionalidades implementadas

### Sistema de evasión de colisiones

- **Lifters random**: Cuando detectan otro lifter cercano, cambian automáticamente su dirección para evitar colisiones
- **Área de tirar basura**: Las colisiones solo están permitidas en el área central (posición entre -10 y 10 en X y Z), donde los lifters pueden amontonarse
- **Fuera del área de basura**: Los lifters random evitan activamente chocar entre sí

### Optimización para método planeado

- Los lifters con método "planned" se detienen menos tiempo en el área de tirar basura (bajan la plataforma 2x más rápido)

### Controles durante la simulación

- **Flecha Izquierda**: Rotar cámara a la izquierda
- **Flecha Derecha**: Rotar cámara a la derecha
- **ESC**: Salir de la simulación

## Archivos generados

Los datos de la simulación se guardan automáticamente en la carpeta `datos/` con el formato:

```
YYYYMMDD_HHMMSS_agent{N}_[method].csv
```

Cada archivo contiene información sobre:

- Timestamp
- ID del agente
- Método usado (planned/random)
- Estado actual
- Posición (x, y, z)
- Nodo actual y siguiente
- Ruta (si es planned)

## Documentación completa del proyecto
[![Abrir PDF](https://img.shields.io/badge/PDF-Abrir-red)](./E1_ActividadIntegradora.pdf)

## Video de simulación en OpenGL
[![Ver Video](https://img.youtube.com/vi/ID_DEL_VIDEO/0.jpg)]([https://www.youtube.com/watch?v=ID_DEL_VIDEO](https://youtu.be/CnlFNbfakMo))
