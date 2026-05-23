import heapq
import math

# =====================================================================
# 1. BASE DE CONOCIMIENTO: Red de estaciones de transporte local
# Modelamos las estaciones como reglas lógicas de conexión y coordenadas (X, Y)
# =====================================================================
MAPA_TRANSPORTE = {
    'Portal Norte':      {'conexiones': {'Toberin': 4, 'Calle 100': 15}, 'pos': (0, 12)},
    'Toberin':           {'conexiones': {'Portal Norte': 4, 'Calle 100': 9}, 'pos': (0, 10)},
    'Calle 100':         {'conexiones': {'Toberin': 9, 'Portal Norte': 15, 'Heroes': 6, 'Calle 72': 12}, 'pos': (0, 7)},
    'Heroes':            {'conexiones': {'Calle 100': 6, 'Calle 72': 5}, 'pos': (0, 5)},
    'Calle 72':          {'conexiones': {'Heroes': 5, 'Calle 100': 12, 'Marly': 4, 'Calle 26': 10}, 'pos': (0, 3)},
    'Marly':             {'conexiones': {'Calle 72': 4, 'Calle 26': 5}, 'pos': (0, 1)},
    'Calle 26':          {'conexiones': {'Marly': 5, 'Calle 72': 10, 'Aguas': 6}, 'pos': (0, 0)},
    'Aguas':             {'conexiones': {'Calle 26': 6}, 'pos': (2, -1)}
}

# =====================================================================
# 2. HEURÍSTICA: Estimación de distancia en línea recta
# Función inteligente para calcular la distancia geométrica al destino
# =====================================================================
def calcular_distancia_estimada(estacion_actual, destino):
    coordenadas_actual = MAPA_TRANSPORTE[estacion_actual]['pos']
    coordenadas_destino = MAPA_TRANSPORTE[destino]['pos']
    
    # Aplicamos la fórmula de distancia euclidiana
    distancia_x = coordenadas_actual[0] - coordenadas_destino[0]
    distancia_y = coordenadas_actual[1] - coordenadas_destino[1]
    return math.sqrt(distancia_x**2 + distancia_y**2)

# =====================================================================
# 3. MOTOR DE BÚSQUEDA HEURÍSTICA (Algoritmo A*)
# Encuentra la ruta más rápida combinando tiempo real y estimación
# =====================================================================
def encontrar_ruta_optima(punto_inicio, punto_destino):
    # Validación de seguridad por si se ingresa una estación errónea
    if punto_inicio not in MAPA_TRANSPORTE or punto_destino not in MAPA_TRANSPORTE:
        return None, "Error: Una de las estaciones no se encuentra registrada en el sistema."

    # Inicializamos la cola de prioridad: (peso_estimado, tiempo_real, nodo_actual, ruta_recorrida)
    estimacion_inicial = calcular_distancia_estimada(punto_inicio, punto_destino)
    frontera_busqueda = [(estimacion_inicial, 0, punto_inicio, [punto_inicio])]
    estaciones_visitadas = set()

    while frontera_busqueda:
        # Evaluamos el nodo con el menor costo f(n) = g(n) + h(n)
        _, tiempo_acumulado, actual, ruta_actual = heapq.heappop(frontera_busqueda)

        if actual in estaciones_visitadas:
            continue
        estaciones_visitadas.add(actual)

        # Si llegamos al destino, retornamos el resultado
        if actual == punto_destino:
            return tiempo_acumulado, ruta_actual

        # Analizamos las conexiones lógicas de la estación actual
        for vecino, tiempo_tramo in MAPA_TRANSPORTE[actual]['conexiones'].items():
            if vecino not in estaciones_visitadas:
                nuevo_tiempo_real = tiempo_acumulado + tiempo_tramo
                peso_total_estimado = nuevo_tiempo_real + calcular_distancia_estimada(vecino, punto_destino)
                
                heapq.heappush(
                    frontera_busqueda, 
                    (peso_total_estimado, nuevo_tiempo_real, vecino, ruta_actual + [vecino])
                )

    return None, "No se logró consolidar una ruta válida entre los puntos."

# =====================================================================
# Bloque de prueba de la estudiante
# =====================================================================
if __name__ == "__main__":
    estacion_origen = 'Portal Norte'
    estacion_destino = 'Aguas'
    
    minutos_totales, itinerario = encontrar_ruta_optima(estacion_origen, estacion_destino)
    
    print("--------------------------------------------------")
    print("     PLANIFICADOR INTELIGENTE DE RUTAS (TM)")
    print("--------------------------------------------------")
    if minutos_totales:
        print(f"Punto de partida: {estacion_origen}")
        print(f"Destino final:   {estacion_destino}\n")
        print(f"▶ Recorrido sugerido: {' ➔ '.join(itinerario)}")
        print(f"⏱ Tiempo total estimado de viaje: {minutos_totales} minutos")
    else:
        print(f"⚠ Alerta del sistema: {itinerario}")
    print("--------------------------------------------------")