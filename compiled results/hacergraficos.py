import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar CSV
df = pd.read_csv("resultados.csv")

# Separar 'color' y las demás funciones
df_color = df[df['fitness'] == 'color']
df_otros = df[df['fitness'] != 'color']

# ===========================
# Curvas de convergencia
# ===========================

# Para color
plt.figure(figsize=(8,5))
for mut_prob, group in df_color.groupby('prob_mutacion'):
    plt.plot(group['t_convergencia_medio'], group['mejor_medio'], marker='o', label=f"Mutación {mut_prob}")
plt.xlabel("Tiempo de convergencia promedio (s)")
plt.ylabel("Mejor aptitud promedio")
plt.title("Curvas de convergencia - Fitness: color")
plt.legend()
plt.grid(True)
plt.show()

# Para otras funciones
plt.figure(figsize=(8,5))
for fitness_name, group in df_otros.groupby("fitness"):
    plt.plot(group['t_convergencia_medio'], group['mejor_medio'], marker='o', label=fitness_name)
plt.xlabel("Tiempo de convergencia promedio (s)")
plt.ylabel("Mejor aptitud promedio")
plt.title("Curvas de convergencia - Todas las funciones excepto 'color'")
plt.legend()
plt.grid(True)
plt.show()

# ===========================
# Varianza entre ejecuciones
# ===========================

# Para color
plt.figure(figsize=(8,5))
sns.barplot(x="fitness", y="desv_media", hue="prob_mutacion", data=df_color)
plt.ylabel("Desviación estándar de aptitud")
plt.xlabel("Función de aptitud")
plt.title("Varianza de aptitud - Fitness: color")
plt.show()

# Para otras funciones
plt.figure(figsize=(8,5))
sns.barplot(x="fitness", y="desv_media", hue="prob_mutacion", data=df_otros)
plt.ylabel("Desviación estándar de aptitud")
plt.xlabel("Función de aptitud")
plt.title("Varianza de aptitud - Todas las funciones excepto 'color'")
plt.show()

# ===========================
# Estabilidad de los algoritmos
# ===========================

# Para color
plt.figure(figsize=(8,5))
sns.scatterplot(
    x="t_convergencia_medio", 
    y="desv_media", 
    hue="fitness", 
    style="prob_mutacion", 
    s=100,
    data=df_color
)
plt.xlabel("Tiempo de convergencia promedio (s)")
plt.ylabel("Desviación estándar de aptitud")
plt.title("Estabilidad del algoritmo - Fitness: color")
plt.grid(True)
plt.show()

# Para otras funciones
plt.figure(figsize=(8,5))
sns.scatterplot(
    x="t_convergencia_medio", 
    y="desv_media", 
    hue="fitness", 
    style="prob_mutacion", 
    s=100,
    data=df_otros
)
plt.xlabel("Tiempo de convergencia promedio (s)")
plt.ylabel("Desviación estándar de aptitud")
plt.title("Estabilidad del algoritmo - Todas las funciones excepto 'color'")
plt.grid(True)
plt.show()
