"""Genetic Algorithm for TCP on a permutation encoding (Section 2.5.2, 3.3.2.1).

Each chromosome is a permutation of test-case indices (the execution order).
Fitness is supplied by the caller so the same engine serves the proposed
method and the agile-unaware baselines (Table 3.6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

FitnessFn = Callable[[np.ndarray], float]


@dataclass
class GAConfig:
    """Table 3.5, p.59-60."""

    population_size: int = 50
    max_iterations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    tournament_size: int = 3
    elitism: int = 1


def _init_population(n_tests: int, pop_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    return [rng.permutation(n_tests) for _ in range(pop_size)]


def _tournament_select(
    population: list[np.ndarray], fitnesses: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    contenders = rng.integers(0, len(population), size=k)
    winner = contenders[np.argmax(fitnesses[contenders])]
    return population[winner]


def order_crossover(parent1: np.ndarray, parent2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """OX1 order crossover: preserves a contiguous slice from parent1 and
    fills the remaining positions with parent2's genes in their relative
    order, skipping genes already present.
    """
    n = len(parent1)
    i, j = sorted(rng.integers(0, n, size=2))
    child = np.full(n, -1, dtype=parent1.dtype)
    child[i:j] = parent1[i:j]
    taken = set(child[i:j].tolist())

    fill_values = [gene for gene in parent2 if gene not in taken]
    fill_positions = [p for p in range(n) if not (i <= p < j)]
    for pos, gene in zip(fill_positions, fill_values):
        child[pos] = gene
    return child


def swap_mutation(chromosome: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Independently, with probability `rate`, swap each position with a
    randomly chosen other position (standard swap mutation for permutations).
    """
    child = chromosome.copy()
    n = len(child)
    for pos in range(n):
        if rng.random() < rate:
            other = rng.integers(0, n)
            child[pos], child[other] = child[other], child[pos]
    return child


def run_ga(
    n_tests: int,
    fitness_fn: FitnessFn,
    config: GAConfig = GAConfig(),
    rng: np.random.Generator | None = None,
    seed_population: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, float, list[float]]:
    """Run the GA to completion.

    Returns (best_ordering, best_fitness, best_fitness_per_generation).
    """
    rng = rng if rng is not None else np.random.default_rng()

    if seed_population is not None:
        population = [p.copy() for p in seed_population]
        while len(population) < config.population_size:
            population.append(rng.permutation(n_tests))
        population = population[: config.population_size]
    else:
        population = _init_population(n_tests, config.population_size, rng)

    history: list[float] = []
    best_ordering: np.ndarray | None = None
    best_fitness = -np.inf

    for _ in range(config.max_iterations):
        fitnesses = np.array([fitness_fn(ind) for ind in population])

        gen_best_idx = int(np.argmax(fitnesses))
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = float(fitnesses[gen_best_idx])
            best_ordering = population[gen_best_idx].copy()
        history.append(best_fitness)

        elite_idx = np.argsort(fitnesses)[::-1][: config.elitism]
        next_population = [population[i].copy() for i in elite_idx]

        while len(next_population) < config.population_size:
            p1 = _tournament_select(population, fitnesses, config.tournament_size, rng)
            p2 = _tournament_select(population, fitnesses, config.tournament_size, rng)
            if rng.random() < config.crossover_rate:
                child = order_crossover(p1, p2, rng)
            else:
                child = p1.copy()
            child = swap_mutation(child, config.mutation_rate, rng)
            next_population.append(child)

        population = next_population

    assert best_ordering is not None
    return best_ordering, best_fitness, history
