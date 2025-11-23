import argparse, datetime, LIB_TC2008B
import json
import os

def load_config_from_json(json_file):
    """Carga la configuración desde un archivo JSON"""
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"El archivo JSON no existe: {json_file}")
    
    with open(json_file, 'r') as f:
        config = json.load(f)
    
    # Crear un objeto similar a argparse.Namespace
    class Config:
        pass
    
    options = Config()
    
    # Mapear los nombres del JSON a los nombres de los argumentos
    # El JSON puede usar nombres más legibles, los convertimos a los nombres del parser
    json_to_arg_map = {
        'lifters': 'lifters',
        'basuras': 'Basuras',
        'basura': 'Basuras',  # Alias alternativo
        'delta': 'Delta',
        'theta': 'theta',
        'radius': 'radious',
        'radious': 'radious',
        'method': 'method',
        'tmax': 'Tmax',
        'Tmax': 'Tmax',
        'm': 'M',
        'M': 'M',
        'matrix_size': 'M',
        'resumen': 'resumen',
        'resumen': 'resumen'
    }
    
    # Asignar valores del JSON al objeto Options
    for json_key, arg_name in json_to_arg_map.items():
        if json_key in config:
            setattr(options, arg_name, config[json_key])
    
    # Agregar el comando si está en el JSON
    if 'command' in config:
        options.command = config['command']
    elif 'simulacion' in config or 'simulation' in config:
        options.command = 'Simulacion'
    
    return options

def merge_options(json_options, cli_options):
    """Combina opciones del JSON con opciones de línea de comandos.
    Los argumentos de CLI tienen prioridad sobre los del JSON."""
    # Crear un objeto que combine ambos
    class MergedOptions:
        pass
    
    merged = MergedOptions()
    
    # Primero copiar valores del JSON
    if json_options:
        for attr in dir(json_options):
            if not attr.startswith('_') and not callable(getattr(json_options, attr)):
                setattr(merged, attr, getattr(json_options, attr))
    
    # Luego sobrescribir con valores de CLI (solo si no son None)
    if cli_options:
        for attr in dir(cli_options):
            if not attr.startswith('_') and not callable(getattr(cli_options, attr)):
                value = getattr(cli_options, attr)
                # Solo sobrescribir si el valor no es None
                # Ignorar atributos especiales como 'config', 'command', 'func'
                if value is not None and attr not in ['config']:
                    setattr(merged, attr, value)
    
    return merged

def main():
    parser = argparse.ArgumentParser("TC2008B Base reto", description = "Base del reto");
    
    subparsers = parser.add_subparsers(dest='command');
    
    subparser = subparsers.add_parser("Simulacion",  description = "Corre simulacion");
    # Agregar argumento para archivo JSON en el subparser
    subparser.add_argument("--config", "--json", type=str, 
                       help="Archivo JSON con la configuración de la simulación");
    subparser.add_argument("--lifters", required = False, type = int, help = "Numero de montacargas");
    subparser.add_argument("--Basuras", required = False, type = int, help = "Numero de basuras");            
    subparser.add_argument("--Delta", required = False, type = float, default = None, help = "Velocidad de simulacion");
    subparser.add_argument("--theta", required = False, type = float, default = None, help = "");    
    subparser.add_argument("--radious", required = False, type = float, default = None, help = "");
    subparser.add_argument("--method", required = False, type = str, default = None, 
                         choices = ["planned", "random"], help = "Método de navegación: planned o random");
    subparser.add_argument("--Tmax", required = False, type = float, default = None, 
                         help = "Duración máxima de la simulación en segundos (por defecto: 60)");
    subparser.add_argument("--M", required = False, type = int, default = None, 
                         help = "Tamaño de la matriz MxM para método random(por defecto: 5)");
    subparser.add_argument("--resumen", required=False, type=str, default=None,
                     choices=["s", "n"], help="Mostrar resumen al final (s/n)")
    subparser.set_defaults(func = LIB_TC2008B.Simulacion);

    subparser = subparsers.add_parser("Nodos",  description = "Genera los nodos de la simulacion");
    subparser.add_argument("--NumeroNodos", required = False, type = int, help = "Numero de nodos");
    subparser.set_defaults(func = LIB_TC2008B.GeneracionDeNodos);
    
    args = parser.parse_args();
    
    # Cargar configuración del JSON si se proporciona
    json_options = None
    if hasattr(args, 'config') and args.config:
        try:
            json_options = load_config_from_json(args.config)
            print(f"Configuración cargada desde: {args.config}")
        except Exception as e:
            print(f"Error al cargar archivo JSON: {e}")
            return
    
    # Si hay JSON y no se especificó comando en CLI, usar el del JSON
    if json_options and (not hasattr(args, 'command') or args.command is None):
        if hasattr(json_options, 'command'):
            args.command = json_options.command
    
    # Combinar opciones (CLI tiene prioridad)
    if json_options:
        Options = merge_options(json_options, args)
        # Asegurar que el comando esté establecido (CLI tiene prioridad)
        if hasattr(args, 'command') and args.command:
            Options.command = args.command
        elif hasattr(json_options, 'command'):
            Options.command = json_options.command
    else:
        Options = args
    
    # Establecer la función según el comando (tanto para JSON como CLI)
    # El comando debería estar en args.command cuando se llama desde CLI
    if not hasattr(Options, 'command') or Options.command is None:
        if hasattr(args, 'command') and args.command:
            Options.command = args.command
    
    if hasattr(Options, 'command') and Options.command:
        if Options.command == 'Simulacion':
            Options.func = LIB_TC2008B.Simulacion
        elif Options.command == 'Nodos':
            Options.func = LIB_TC2008B.GeneracionDeNodos
    
    # Aplicar valores por defecto si no están definidos
    if not hasattr(Options, 'Delta') or Options.Delta is None:
        Options.Delta = 0.05
    if not hasattr(Options, 'theta') or Options.theta is None:
        Options.theta = 0
    if not hasattr(Options, 'radious') or Options.radious is None:
        Options.radious = 30
    if not hasattr(Options, 'method') or Options.method is None:
        Options.method = "planned"
    if not hasattr(Options, 'Tmax') or Options.Tmax is None:
        Options.Tmax = 60.0
    if not hasattr(Options, 'M') or Options.M is None:
        Options.M = 5
    if not hasattr(Options, 'resumen') or Options.resumen is None:
        Options.resumen = "s"
    
    # Validar argumentos requeridos
    if Options.command == "Simulacion":
        if not hasattr(Options, 'lifters') or Options.lifters is None:
            if json_options and hasattr(json_options, 'lifters'):
                Options.lifters = json_options.lifters
            else:
                parser.error("--lifters es requerido (o debe estar en el archivo JSON)")
        if not hasattr(Options, 'Basuras') or Options.Basuras is None:
            if json_options and hasattr(json_options, 'Basuras'):
                Options.Basuras = json_options.Basuras
            else:
                parser.error("--Basuras es requerido (o debe estar en el archivo JSON)")
    
    # Imprimir configuración de forma más legible
    print("Configuración de la simulación:")
    if hasattr(Options, 'command'):
        print(f"  Comando: {Options.command}")
    if hasattr(Options, 'lifters'):
        print(f"  Lifters: {Options.lifters}")
    if hasattr(Options, 'Basuras'):
        print(f"  Basuras: {Options.Basuras}")
    if hasattr(Options, 'method'):
        print(f"  Método: {Options.method}")
    if hasattr(Options, 'M'):
        print(f"  Matriz MxM: {Options.M}")
    if hasattr(Options, 'Tmax'):
        print(f"  Tmax: {Options.Tmax}")
    print()

    if hasattr(Options, 'func') and Options.func:
        Options.func(Options)
    else:
        print("Error: No se pudo determinar la función a ejecutar.")
        parser.print_help() 

if __name__ == "__main__":
    print("\n" + "\033[0;32m" + "[start] " + str(datetime.datetime.now()) + "\033[0m" + "\n");
    main();
    print("\n" + "\033[0;32m" + "[end] "+ str(datetime.datetime.now()) + "\033[0m" + "\n");