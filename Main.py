import argparse, datetime, LIB_TC2008B

def main():
    parser = argparse.ArgumentParser("TC2008B Base reto", description = "Base del reto");
    subparsers = parser.add_subparsers();
    
    subparser = subparsers.add_parser("Simulacion",  description = "Corre simulacion");
    subparser.add_argument("--lifters", required = True, type = int, help = "Numero de montacargas");
    subparser.add_argument("--Basuras", required = True, type = int, help = "Numero de basuras");            
    subparser.add_argument("--Delta", required = False, type = float, default = 0.1, help = "Velocidad de simulacion (aumentado por defecto para mayor rapidez)");
    subparser.add_argument("--theta", required = False, type = float, default = 0, help = "");    
    subparser.add_argument("--radious", required = False, type = float, default = 30, help = "");
    subparser.add_argument("--method", required = False, type = str, default = "planned", 
                         choices = ["planned", "random"], help = "Método de navegación para todos los lifters: planned o random (si no se usa --methods)");
    subparser.add_argument("--methods", required = False, type = str, default = None,
                         help = "Métodos individuales por lifter separados por comas (ej: 'planned,random,planned'). Debe tener tantos métodos como lifters");
    subparser.add_argument("--Tmax", required = False, type = float, default = 60.0, 
                         help = "Duración máxima de la simulación en segundos (por defecto: 60)");
    subparser.add_argument("--M", required = False, type = int, default = 5, 
                         help = "Tamaño de la matriz MxM (afecta la densidad de nodos en el mapa). Por defecto: 5. Valores mayores = más nodos más juntos, valores menores = menos nodos más separados");
    subparser.add_argument("--resumen", required=False, type=str, default="s",
                     choices=["s", "n"], help="Mostrar resumen al final (s/n)")
    subparser.set_defaults(func = LIB_TC2008B.Simulacion);

    subparser = subparsers.add_parser("Nodos",  description = "Genera los nodos de la simulacion");
    subparser.add_argument("--NumeroNodos", required = True, type = int, help = "Numero de nodos");
    subparser.set_defaults(func = LIB_TC2008B.GeneracionDeNodos);
    
    Options = parser.parse_args();
    
    print(str(Options) + "\n");

    if hasattr(Options, 'func'):
        Options.func(Options)
    else:
        parser.print_help() 

if __name__ == "__main__":
    print("\n" + "\033[0;32m" + "[start] " + str(datetime.datetime.now()) + "\033[0m" + "\n");
    main();
    print("\n" + "\033[0;32m" + "[end] "+ str(datetime.datetime.now()) + "\033[0m" + "\n");