import time
import numpy as np


# Artificial Orcas Algorithm (AOA)
def AOA(population, fitness_function, fmin, fmax, max_iter):
    npop, dim = population.shape
    num_pods = 2
    Dc = 1.0
    Dp = 0.5
    Dr = 0.1
    fitness = fitness_function(population)
    Convergence_curve = np.zeros(max_iter)
    ct = time.time()

    # AOA Main Loop
    for t in range(max_iter):
        # Step 3a: Sort the population by fitness
        sorted_indices = np.argsort(fitness)
        population = population[sorted_indices]
        fitness = fitness[sorted_indices]

        # Intensification Search (Echolocation/Cooperation)
        best_orca = population[0]
        for i in range(npop):
            if np.random.rand() < 0.5:
                # Echolocation search
                population[i] = population[i] + 0.5 * (best_orca - population[i]) * np.random.rand(dim)
            else:
                # Cooperation search (local exploration)
                population[i] = population[i] + np.random.uniform(-1, 1, dim)

        # Step 3b: Recalculate fitness after intensification search
        fitness = fitness_function(population)

        # Diversification Search (update worst orcas in population)
        worst_orcas = sorted_indices[-num_pods:]
        for idx in worst_orcas:
            population[idx] = population[idx] + Dr * np.random.uniform(-1, 1, dim)

        # Step 3c: Recalculate fitness after diversification
        fitness = fitness_function(population)

        best_fitness = np.min(fitness)
        Convergence_curve[t] = best_fitness
    # Step 5: Return the best solution and its fitness
    best_fitness = np.min(fitness)
    best_orca = population[np.argmin(fitness)]
    CT = time.time() - ct
    return best_fitness, Convergence_curve, best_orca, CT