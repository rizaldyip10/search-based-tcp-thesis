"""Agile-aware hybrid GA-CSO (Section 3.3.2.3, p.58-59) -- the thesis's
proposed artifact (RO2).

Each generation performs:
    1. a GA phase (tournament selection, order crossover, swap mutation),
    2. a CSO phase (Lévy-flight perturbation around random/best nests),
    3. fitness evaluation using the caller-supplied fitness function.

`run_single_sprint` runs this on one static TestSuite (used for the
agile-unaware "Hybrid GA-CSO (no agile factors)" baseline and for testing).

`run_across_sprints` runs it across a *sequence* of TestSuite snapshots
(one per simulated Scrum sprint, e.g. Defects4J commits mapped to sprints,
Section 3.4.1.2), carrying the population forward between sprints and
applying the sprint re-initialisation mechanism: at each sprint boundary,
`reinit_rate` (default 25%, matching CSO's pa -- Table 3.5) of the
population is reseeded with orderings that bias newly added/modified test
cases toward the front, preventing the convergence decay that static TCP
methods suffer under volatile agile suites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from tcp.cso import CSOConfig, _levy_perturb
from tcp.ga import GAConfig, _tournament_select, order_crossover, swap_mutation
from tcp.suite import TestSuite

FitnessFn = Callable[[np.ndarray], float]


@dataclass
class HybridConfig:
    """Combines GA and CSO parameters (Table 3.5, p.59-60) plus the
    sprint re-initialisation rate, which matches CSO's abandoned-nest
    fraction pa by default (p.59, "Matches pa; sensitivity-tested in Phase 3").
    """

    population_size: int = 50
    max_iterations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    tournament_size: int = 3
    elitism: int = 1
    pa: float = 0.25
    levy_lambda: float = 1.5
    reinit_rate: float = 0.25

    def ga_config(self) -> GAConfig:
        return GAConfig(
            population_size=self.population_size,
            max_iterations=self.max_iterations,
            crossover_rate=self.crossover_rate,
            mutation_rate=self.mutation_rate,
            tournament_size=self.tournament_size,
            elitism=self.elitism,
        )

    def cso_config(self) -> CSOConfig:
        return CSOConfig(
            population_size=self.population_size,
            max_iterations=self.max_iterations,
            pa=self.pa,
            levy_lambda=self.levy_lambda,
        )


def _ga_phase(
    population: list[np.ndarray],
    fitnesses: np.ndarray,
    config: HybridConfig,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    elite_idx = np.argsort(fitnesses)[::-1][: config.elitism]
    next_population = [population[i].copy() for i in elite_idx]
    while len(next_population) < len(population):
        p1 = _tournament_select(population, fitnesses, config.tournament_size, rng)
        p2 = _tournament_select(population, fitnesses, config.tournament_size, rng)
        child = order_crossover(p1, p2, rng) if rng.random() < config.crossover_rate else p1.copy()
        child = swap_mutation(child, config.mutation_rate, rng)
        next_population.append(child)
    return next_population


def _cso_phase(
    population: list[np.ndarray],
    fitnesses: np.ndarray,
    best_ordering: np.ndarray,
    config: HybridConfig,
    fitness_fn: FitnessFn,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], np.ndarray]:
    population = [ind.copy() for ind in population]
    fitnesses = fitnesses.copy()

    for i in range(len(population)):
        source = best_ordering if rng.random() < 0.5 else population[i]
        candidate = _levy_perturb(source, config.levy_lambda, rng)
        candidate_fitness = fitness_fn(candidate)
        if candidate_fitness > fitnesses[i]:
            population[i] = candidate
            fitnesses[i] = candidate_fitness

    n_abandon = max(1, int(round(config.pa * len(population))))
    worst_idx = np.argsort(fitnesses)[:n_abandon]
    n_tests = len(population[0])
    for idx in worst_idx:
        population[idx] = rng.permutation(n_tests)
        fitnesses[idx] = fitness_fn(population[idx])

    return population, fitnesses


def run_single_sprint(
    n_tests: int,
    fitness_fn: FitnessFn,
    config: HybridConfig = HybridConfig(),
    rng: np.random.Generator | None = None,
    seed_population: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, float, list[float], list[np.ndarray]]:
    """Run the hybrid GA-CSO on one static suite.

    Returns (best_ordering, best_fitness, history, final_population) -- the
    final population is exposed so callers can warm-start the next sprint.
    """
    rng = rng if rng is not None else np.random.default_rng()

    if seed_population is not None:
        population = [p.copy() for p in seed_population]
        while len(population) < config.population_size:
            population.append(rng.permutation(n_tests))
        population = population[: config.population_size]
    else:
        population = [rng.permutation(n_tests) for _ in range(config.population_size)]

    fitnesses = np.array([fitness_fn(ind) for ind in population])
    best_idx = int(np.argmax(fitnesses))
    best_ordering = population[best_idx].copy()
    best_fitness = float(fitnesses[best_idx])
    history: list[float] = []

    for _ in range(config.max_iterations):
        population = _ga_phase(population, fitnesses, config, rng)
        fitnesses = np.array([fitness_fn(ind) for ind in population])

        population, fitnesses = _cso_phase(
            population, fitnesses, best_ordering, config, fitness_fn, rng
        )

        gen_best_idx = int(np.argmax(fitnesses))
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = float(fitnesses[gen_best_idx])
            best_ordering = population[gen_best_idx].copy()
        history.append(best_fitness)

    return best_ordering, best_fitness, history, population


def priority_biased_permutation(
    n_tests: int, priority_mask: np.ndarray, rng: np.random.Generator, boost: float = 1.0
) -> np.ndarray:
    """A random permutation biased to place `priority_mask`-marked tests
    earlier on average: each test gets a random score, boosted for
    new/modified tests, and the ordering sorts by descending score. This
    implements "solutions that give priority to newly added or modified
    test cases" (p.58) while keeping the rest of the ordering randomised for
    population diversity.
    """
    scores = rng.random(n_tests) + np.where(priority_mask, boost, 0.0)
    return np.argsort(-scores)


def _reinitialise_for_new_sprint(
    population: list[np.ndarray],
    fitnesses: np.ndarray,
    suite: TestSuite,
    reinit_rate: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Sprint re-initialisation (p.58): replace the worst `reinit_rate`
    fraction of the carried-over population with priority-biased orderings
    for the new sprint's test suite.
    """
    n_tests = suite.n_tests
    population = [ind.copy() for ind in population]

    # Carried-over individuals may reference stale indices if n_tests
    # changed between sprints; drop any that no longer fit.
    population = [ind for ind in population if len(ind) == n_tests and ind.max() < n_tests]
    if not population:
        population = [rng.permutation(n_tests) for _ in range(len(fitnesses))]
        fitnesses = np.full(len(population), -np.inf)

    n_reinit = max(1, int(round(reinit_rate * len(population))))
    worst_idx = np.argsort(fitnesses[: len(population)])[:n_reinit]
    for idx in worst_idx:
        population[idx] = priority_biased_permutation(n_tests, suite.is_new_or_modified, rng)
    return population


def run_across_sprints(
    suites: Sequence[TestSuite],
    fitness_fn_factory: Callable[[TestSuite], FitnessFn],
    config: HybridConfig = HybridConfig(),
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Run the agile-aware hybrid across a sequence of per-sprint TestSuite
    snapshots, carrying the population forward and applying sprint
    re-initialisation at each boundary (sprints after the first).

    Returns one result dict per sprint: {"ordering", "fitness", "history"}.
    """
    rng = rng if rng is not None else np.random.default_rng()
    results: list[dict] = []
    population: list[np.ndarray] | None = None

    for sprint_idx, suite in enumerate(suites):
        fitness_fn = fitness_fn_factory(suite)

        seed = None
        if population is not None:
            fitnesses = np.array(
                [
                    fitness_fn(ind) if len(ind) == suite.n_tests else -np.inf
                    for ind in population
                ]
            )
            seed = _reinitialise_for_new_sprint(
                population, fitnesses, suite, config.reinit_rate, rng
            )

        best_ordering, best_fitness, history, population = run_single_sprint(
            suite.n_tests, fitness_fn, config, rng, seed_population=seed
        )
        results.append(
            {
                "sprint": sprint_idx,
                "ordering": best_ordering,
                "fitness": best_fitness,
                "history": history,
            }
        )

    return results
