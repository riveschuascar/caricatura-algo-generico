import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
import time
import random

class TransformationParams:
    def __init__(self, params=None):
        if params:
            self.params = params
        else:
            self.params = {
                'brightness': np.random.uniform(-50, 50),
                'contrast': np.random.uniform(0.5, 1.5),
                'warp_scale': np.random.uniform(0, 20), 
            }

def apply_transformations(image, individual):
    img_transformed = image.copy()
    
    brightness = individual.params['brightness']
    contrast = individual.params['contrast']
    img_transformed = cv2.addWeighted(img_transformed, contrast, np.zeros_like(img_transformed), 0, brightness)
    
    warp_scale = individual.params['warp_scale']
    if warp_scale > 0:
        rows, cols, _ = img_transformed.shape
        noise = np.random.randn(rows, cols, 2) * warp_scale
        map_x = (np.repeat(np.arange(cols), rows).reshape(cols, rows).T + noise[:,:,0]).astype(np.float32)
        map_y = (np.repeat(np.arange(rows), cols).reshape(rows, cols) + noise[:,:,1]).astype(np.float32)
        img_transformed = cv2.remap(img_transformed, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    return np.clip(img_transformed, 0, 255).astype(np.uint8)


def fitness_ssim(generated_img, target_img):
    generated_gray = cv2.cvtColor(generated_img, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    return ssim(generated_gray, target_gray)

def fitness_edge_difference(generated_img, original_img):
    original_edges = cv2.Canny(original_img, 100, 200)
    generated_edges = cv2.Canny(generated_img, 100, 200)
    return np.sum(generated_edges > 0) / (original_img.shape[0] * original_img.shape[1])

def fitness_color_variance(generated_img):
    return np.var(generated_img, axis=(0, 1)).mean()

def fitness_ssim_edge(generated_img, original_img, target_img):
    generated_gray = cv2.cvtColor(generated_img, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    ssim_score = ssim(generated_gray, target_gray)

    original_edges = cv2.Canny(original_img, 100, 200)
    generated_edges = cv2.Canny(generated_img, 100, 200)
    edge_score = np.sum(generated_edges > 0) / (original_img.shape[0] * original_img.shape[1])

    return (0.3 * ssim_score) + (0.7 * edge_score)

class GeneticAlgorithm:
    def __init__(self, population_size, generations, mutation_rate, crossover_rate, original_image, target_image, fitness_func_name):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.original_image = original_image
        self.target_image = target_image
        self.population = [TransformationParams() for _ in range(population_size)]
        
        # Mapeo para seleccionar la función de aptitud
        self.fitness_functions = {
            "ssim": lambda img: fitness_ssim(img, self.target_image),
            "edge": lambda img: fitness_edge_difference(img, self.original_image),
            "color": lambda img: fitness_color_variance(img),
            "ssim_edge": lambda img: fitness_ssim_edge(img, self.original_image, self.target_image),
        }
        self.fitness_func = self.fitness_functions[fitness_func_name]

    def calculate_fitness(self, individual):
        transformed_img = apply_transformations(self.original_image, individual)
        return self.fitness_func(transformed_img)

    def select(self, fitnesses):
        tournament_size = 5
        selected = []
        for _ in range(self.population_size):
            participants_indices = np.random.choice(range(self.population_size), tournament_size, replace=False)
            participants_fitness = [fitnesses[i] for i in participants_indices]
            winner_index = participants_indices[np.argmax(participants_fitness)]
            selected.append(self.population[winner_index])
        return selected

    def crossover(self, parent1, parent2):
        child1_params, child2_params = {}, {}
        if random.random() < self.crossover_rate:
            keys = list(parent1.params.keys())
            crossover_point = random.randint(1, len(keys) - 1)
            for i, key in enumerate(keys):
                if i < crossover_point:
                    child1_params[key] = parent1.params[key]
                    child2_params[key] = parent2.params[key]
                else:
                    child1_params[key] = parent2.params[key]
                    child2_params[key] = parent1.params[key]
            return TransformationParams(child1_params), TransformationParams(child2_params)
        return parent1, parent2
        
    def mutate(self, individual):
        mutated_params = individual.params.copy()
        for key in mutated_params:
            if random.random() < self.mutation_rate:
                if key == 'brightness':
                    mutated_params[key] += np.random.normal(0, 10)
                elif key == 'contrast':
                    mutated_params[key] += np.random.normal(0, 0.2)
                elif key == 'warp_scale':
                    mutated_params[key] += np.random.normal(0, 2)
        return TransformationParams(mutated_params)

    def run(self):
        best_fitness_history = []
        best_individual = None
        best_fitness = -float('inf')
        start_time = time.time()

        for gen in range(self.generations):
            fitnesses = [self.calculate_fitness(ind) for ind in self.population]

            current_best_idx = np.argmax(fitnesses)
            if fitnesses[current_best_idx] > best_fitness:
                best_fitness = fitnesses[current_best_idx]
                best_individual = self.population[current_best_idx]
            
            best_fitness_history.append(best_fitness)
            print(f"Generación {gen+1}/{self.generations} - Mejor Aptitud: {best_fitness:.4f}")

            selected_population = self.select(fitnesses)

            next_population = []
            for i in range(0, self.population_size, 2):
                parent1, parent2 = selected_population[i], selected_population[i+1]
                child1, child2 = self.crossover(parent1, parent2)
                next_population.append(self.mutate(child1))
                next_population.append(self.mutate(child2))
            
            self.population = next_population
        
        convergence_time = time.time() - start_time
        return best_individual, best_fitness, best_fitness_history, convergence_time


def run_experiment(params_combinations, original_img, target_img, runs_per_combination=10):
    if not os.path.exists("results"):
        os.makedirs("results")

    for i, params in enumerate(params_combinations):
        print(f"\n--- Ejecutando Experimento {i+1}/{len(params_combinations)} con params: {params} ---")
        
        params_dir_name = f"results/fit_{params['fitness_func_name']}_mut_{params['mutation_rate']}"
        if not os.path.exists(params_dir_name):
            os.makedirs(params_dir_name)
        
        all_run_metrics = {"best_fitnesses": [], "convergence_times": []}

        for run in range(runs_per_combination):
            print(f"  Run {run+1}/{runs_per_combination}")
            random.seed(run)
            np.random.seed(run)

            ga = GeneticAlgorithm(
                population_size=50, generations=30,
                mutation_rate=params['mutation_rate'], crossover_rate=0.8,
                original_image=original_img, target_image=target_img,
                fitness_func_name=params['fitness_func_name']
            )
            best_ind, best_fit, _, conv_time = ga.run()
            
            all_run_metrics["best_fitnesses"].append(best_fit)
            all_run_metrics["convergence_times"].append(conv_time)
            
            best_img = apply_transformations(original_img, best_ind)
            cv2.imwrite(os.path.join(params_dir_name, f"best_image_run_{run+1}.png"), best_img)

        with open(os.path.join(params_dir_name, "summary.txt"), "w") as f:
            f.write(f"Resultados para los parámetros: {params}\n")
            f.write(f"Mejor Aptitud Promedio: {np.mean(all_run_metrics['best_fitnesses'])}\n")
            f.write(f"Desv. Est. Aptitud: {np.std(all_run_metrics['best_fitnesses'])}\n")
            f.write(f"Tiempo de Convergencia Promedio: {np.mean(all_run_metrics['convergence_times'])}s\n")

if __name__ == '__main__':
    try:
        original_image = cv2.imread("input_images/rostro.png")
        target_image = cv2.imread("input_images/caricatura_objetivo.webp")
        if original_image is None or target_image is None:
            raise FileNotFoundError("Una o ambas imágenes no se encontraron.")
        original_image = cv2.resize(original_image, (200, 200))
        target_image = cv2.resize(target_image, (200, 200))
    except Exception as e:
        print(f"Error: {e}")
        print("Asegúrate de tener 'rostro.jpeg' y 'caricatura_objetivo.jpeg' en el mismo directorio.")
        exit()

    parameter_combinations = [
        {"fitness_func_name": "ssim", "mutation_rate": 0.05},
        {"fitness_func_name": "ssim", "mutation_rate": 0.2},
        {"fitness_func_name": "edge", "mutation_rate": 0.05},
        {"fitness_func_name": "edge", "mutation_rate": 0.2},
        {"fitness_func_name": "color", "mutation_rate": 0.05},
        {"fitness_func_name": "color", "mutation_rate": 0.2},
        {"fitness_func_name": "ssim_edge", "mutation_rate": 0.05},
        {"fitness_func_name": "ssim_edge", "mutation_rate": 0.2},
    ]
    
    run_experiment(parameter_combinations, original_image, target_image, runs_per_combination=10)
    print("\n¡Experimentos completados! Revisa la carpeta 'results'.")