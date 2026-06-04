import heapq
import math
#sistema inteligente de transporte Masivo
# Lorena Lozano /Ing. Ciencia de Datos.
# =====================================================================# =====================================================================
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
# . 4 Nuevo componente de I.A.
# =====================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def generar_dataset_historico_transporte(num_registros=200):
    """Simulador de un numero de viajes basados en la red de transportes de MAPA_TRANSPORTE.
    Identifica caracteristicas clave, para decidir si un tramo tendrá un retraso crítico  (1) o no (0) """
    np.random.seed(42) #Replicabilidad cientifica
    estaciones=list(MAPA_TRANSPORTE.keys())

    datos=[]
    for _ in range(num_registros):
        #Elegir un tramo aleatorio
        origen=np.random.choice(estaciones)
        posibles_destinos=list(MAPA_TRANSPORTE[origen]['conexiones'].keys())
        destino=np.random.choice(posibles_destinos)

    #(atributos predictores)
        hora=np.random.randint(5,23) #horas del sistema entre 5:am y 11:pm
        es_hora_pico = 1 if (7 <= hora <= 9) or (17 <= hora <= 19) else 0
        clima=np.random.choice(['despejado','lluvia moderada','lluvia fuerte'])
        dia_semana = np.random.choice(['laboral','Fin de semana'])

        #regla subyacente para simular etiqueta "retraso_critico" (saturación del sistema)
        #la probabilidad de retraso aumenta por clima adverso u hora pico.

        probabilidad_retraso =0.1
        if es_hora_pico ==1: probabilidad_retraso +=0.4
        if clima =='lluvia fuerte': probabilidad_retraso +=0.3
        if clima == 'lluvia moderada': probabilidad_retraso += 0.15

        retraso_critico = 1 if np.random.rand() < probabilidad_retraso else 0

        datos.append ({
            'Estacion_origen':origen,
            'Estacion_Destino':destino,
            'Hora_viaje':hora,
            'Es_Hora_Pico':es_hora_pico,
            'Clima':clima,
            'Dia_semana':dia_semana,
            'Retraso_Critico':retraso_critico #Esta es la variable objetivo o la etiqueta
        })

    df = pd.DataFrame(datos)
    return df
def ejecutar_modulo_aprendizaje_supervisado():
        print("\============================================================= " )
        print("\=====  I. A MODULO DE APRENDIZAJE SUPERVISADO   ===============")
        print("\============================================================= " )

        #1.Simulador: obtiene y describe fuentes de datos 
        df_transporte = generar_dataset_historico_transporte(300)
        print(f" ✔ dataset generado exitosamente con {df_transporte.shape[0]} registros historicos")
        print("muestra de los datos identificados: ")
        print(df_transporte[['Estacion_origen', 'Hora_viaje', 'Clima', 'Retraso_Critico']].head())

        df_transporte.to_csv('datos_historicos_transporte.csv',index=False)
        print(" ✔ Archivo('datos_historicos_transporte.csv'exportado)")

        df_modelo =pd.get_dummies(df_transporte, columns=['Estacion_origen', 'Estacion_Destino', 'Clima', 'Dia_semana'])

        X= df_modelo.drop(columns=['Retraso_Critico'])
        Y=df_modelo['Retraso_Critico']

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
        arbol_transporte = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
        arbol_transporte.fit(X_train, y_train)
        print("✔ modelo de árbol de decisión entrenado bajo entropia")

        y_pred = arbol_transporte.predict(X_test)

        print("\n---REPORTE DE PRUEBAS DE COMPONENTE I.A")
        print("Matriz de confusion: ")
        print(confusion_matrix(y_test, y_pred))

        plt.figure(figsize=(16,8))
        plot_tree(arbol_transporte, feature_names=X.columns, class_names=['Sin Retraso', 'Retraso Critico'], filled=True, rounded=True, fontsize=8)
        plt.title("Estructura Jerarquica del árbol de decision - Gestión de retrasos TM", fontsize=14)
        plt.savefig('arbol decision transporte.png', dpi=300, bbox_inches='tight')
        print("✔ gráfico del árbol exportado como 'arbol_decision_transporte.png")
        plt.show()

# =====================================================================
# . 4 Nuevo componente de I.A.
# =====================================================================

if __name__== "__main__":
    #Se ejecuta el algortimo A original
    estacion_origen = 'Portal Norte'
    estacion_destino = 'Aguas'
    minutos_totales, itinerario, = encontrar_ruta_optima(estacion_origen, estacion_destino)

    print("*********************************************************")
    print("********* PLANIFICADOR INTELIGENTE DE RUTAS (TM) ********")
    print("*********************************************************")

    if minutos_totales:
        print(f"Punto de partida:{estacion_origen}")
        print(f"Destino final: {estacion_destino}\n")
        print(f" -> Recorrido sugerido: {' ➔ '.join(itinerario)}")
        print(f"Tiempo total estimado de viaje: {minutos_totales} minutes")
    else:
        print(f"Alerta Del Sistema: {itinerario}")
        print("****************************************************")

#Llamada al nuevo componente:
ejecutar_modulo_aprendizaje_supervisado()
              