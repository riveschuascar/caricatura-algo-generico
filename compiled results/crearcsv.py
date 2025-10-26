import csv
import re
import ast

# Lista de archivos compilados
archivos = [
    "compiled results/compilation_summary_01.txt",
    "compiled results/compilation_summary_02.txt",
    "compiled results/compilation_summary_03.txt",
    "compiled results/compilation_summary_04.txt",
    "compiled results/compilation_summary_05.txt",
    "compiled results/compilation_summary_06.txt",
    "compiled results/compilation_summary_07.txt"
]

output_csv = "resultados.csv"

with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['fitness', 'prob_mutacion', 'desv_media', 'mejor_medio', 't_convergencia_medio'])

    for archivo in archivos:
        with open(archivo, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Cada bloque comienza con "--- Contenido de"
            bloques = re.split(r"--- Contenido de .*? ---", content)
            
            for bloque in bloques:
                bloque = bloque.strip()
                if not bloque:
                    continue
                
                # Extraer parámetros
                params_match = re.search(r"Resultados para los parametros: (.+)", bloque)
                if params_match:
                    params_dict = ast.literal_eval(params_match.group(1))
                    fitness = params_dict.get('fitness_func_name')
                    prob_mutacion = params_dict.get('mutation_rate')
                else:
                    fitness = None
                    prob_mutacion = None

                # Extraer valores de aptitud y tiempo
                mejor_match = re.search(r"Mejor Aptitud Promedio: ([\d\.]+)", bloque)
                mejor_medio = float(mejor_match.group(1)) if mejor_match else None

                desv_match = re.search(r"Desv. Est. Aptitud: ([\d\.]+)", bloque)
                desv_media = float(desv_match.group(1)) if desv_match else None

                tiempo_match = re.search(r"Tiempo de Convergencia Promedio: ([\d\.]+)s", bloque)
                t_convergencia_medio = float(tiempo_match.group(1)) if tiempo_match else None

                # Escribir la fila
                writer.writerow([fitness, prob_mutacion, desv_media, mejor_medio, t_convergencia_medio])

print(f"CSV generado en {output_csv}")
