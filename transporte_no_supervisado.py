import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. SIMULADOR DE DATOS HISTÓRICOS (APRENDIZAJE NO SUPERVISADO)
def generar_datos_clustering():
    np.random.seed(42)
    registros = 300
    
   
    hora_viaje = np.random.randint(5, 24, registros)
    
    afluencia = []
    tiempo_espera = []
    
    for hora in hora_viaje:
        if (7 <= hora <= 9) or (17 <= hora <= 19):  # Horas Pico
            afluencia.append(np.random.randint(800, 1500))
            tiempo_espera.append(np.random.uniform(10, 25))
        elif (10 <= hora <= 16):  # Horas Valle / Medias
            afluencia.append(np.random.randint(300, 799))
            tiempo_espera.append(np.random.uniform(4, 9))
        else:  # Horas Nocturnas o de baja afluencia
            afluencia.append(np.random.randint(50, 299))
            tiempo_espera.append(np.random.uniform(2, 5))
            
    df = pd.DataFrame({
        'Hora_Viaje': hora_viaje,
        'Afluencia_Pasajeros': afluencia,
        'Tiempo_Espera_Minutos': np.round(tiempo_espera, 1)
    })
    return df

print("*********************************************************")
print("===== I. A. MODULO DE APRENDIZAJE NO SUPERVISADO =========")
print("*********************************************************")
df_transporte = generar_datos_clustering()
print("✓ Dataset para clustering generado con 300 registros.")
print("\nMuestra de los datos sin etiquetas (Primeras filas):")
print(df_transporte.head())

df_transporte.to_csv('datos_clustering_transporte.csv', index=False)
print("✓ Archivo 'datos_clustering_transporte.csv' exportado.")


features = ['Afluencia_Pasajeros', 'Tiempo_Espera_Minutos']
X = df_transporte[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_transporte['Cluster'] = kmeans.fit_predict(X_scaled)
print("✓ Algoritmo K-Means entrenado exitosamente con K=3.")


print("\n--- CARACTERÍSTICAS DE LOS CLUSTERS ENCONTRADOS ---")
resumen = df_transporte.groupby('Cluster')[features].mean()
print(resumen)


plt.figure(figsize=(10, 6))
colores = ['orange', 'blue', 'green']
labels = ['Cluster 0', 'Cluster 1', 'Cluster 2']

for i in range(3):
    cluster_data = df_transporte[df_transporte['Cluster'] == i]
    plt.scatter(cluster_data['Afluencia_Pasajeros'], 
                cluster_data['Tiempo_Espera_Minutos'], 
                c=colores[i], label=labels[i], alpha=0.7, edgecolors='k')

plt.title('Agrupamiento K-Means - Segmentación de Congestión en Transporte')
plt.xlabel('Afluencia de Pasajeros (Personas)')
plt.ylabel('Tiempo de Espera (Minutos)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)


plt.savefig('cluster_transporte_kmeans.png', dpi=300)
print("\n✓ Gráfico de dispersión exportado como 'cluster_transporte_kmeans.png'")
plt.show()