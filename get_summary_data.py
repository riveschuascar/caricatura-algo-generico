import os

results_dir = "results"
compiled_dir = "compiled results"

compiled_file_path = os.path.join(compiled_dir, "compilation_summary.txt")

with open(compiled_file_path, "w") as compiled_file:
    for root, dirs, files in os.walk(results_dir):
        if "summary.txt" in files:
            summary_path = os.path.join(root, "summary.txt")
            compiled_file.write(f"--- Contenido de {summary_path} ---\n")
            with open(summary_path, "r") as f:
                compiled_file.write(f.read() + "\n\n")

print(f"Todos los summary.txt han sido recopilados en {compiled_file_path}")
