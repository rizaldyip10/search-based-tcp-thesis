"""Cuckoo Search Optimization (CSO) for TCP on a permutation encoding
(Section 2.5.2, 3.3.2.1; Yang & Deb, 2009).

Lévy flights are continuous-domain moves, so for the permutation/combinatorial
TCP representation we follow the common SBST adaptation: the Lévy step length
(drawn via Mantegna's algorithm) is mapped to a *number of perturbation moves*
(adjacent-swap "shuffles") applied around a reference nest. Large Lévy steps
(heavy-tailed, occasionally huge) translate into large jumps away from the
current nest -- giving the same "mostly-local, occasionally-global" search
behaviour as continuous Lévy flights -- while still landing on a valid
permutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.special import gamma

FitnessFn = Callable[[np.ndarray], float]


@dataclass
class CSOConfig:
    """Table 3.5, p.59-60."""

    population_size: int = 50
    max_iterations: int = 100
    pa: float = 0.25  # abandoned-nest fraction
    levy_lambda: float = 1.5


def _levy_step_length(lam: float, rng: np.random.Generator, size: int = 1) -> np.ndarray:
    """Mantegna's algorithm for sampling Lévy-stable step lengths."""
    num = gamma(1 + lam) * np.sin(np.pi * lam / 2)
    den = gamma((1 + lam) / 2) * lam * 2 ** ((lam - 1) / 2)
    sigma_u = (num / den) ** (1 / lam)
    u = rng.normal(0, sigma_u, size=size)
    v = rng.normal(0, 1, size=size)
    return u / np.abs(v) ** (1 / lam)


def _levy_perturb(nest: np.ndarray, lam: float, rng: np.random.Generator) -> np.ndarray:
    """Perturb `nest` by a number of random swaps proportional to a sampled
    Lévy step length, clipped to a sane range so a single flight can span
    anywhere from a tiny local tweak to a near-total reshuffle.
    """
    n = len(nest)
    step = abs(float(_levy_step_length(lam, rng, size=1).item()))
    n_swaps = int(np.clip(round(step), 1, n))
    child = nest.copy()
    positions = rng.choice(n, size=min(2 * n_swaps, n), replace=False)
    for k in range(0, len(positions) - 1, 2):
        a, b = positions[k], positions[k + 1]
        child[a], child[b] = child[b], child[a]
    return child


def run_cso(
    n_tests: int,
    fitness_fn: FitnessFn,
    config: CSOConfig = CSOConfig(),
    rng: np.random.Generator | None = None,
    seed_population: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, float, list[float]]:
    """Run CSO to completion. Returns (best_ordering, best_fitness, history)."""
    rng = rng if rng is not None else np.random.default_rng()

    if seed_population is not None:
        nests = [p.copy() for p in seed_population]
        while len(nests) < config.population_size:
            nests.append(rng.permutation(n_tests))
        nests = nests[: config.population_size]
    else:
        nests = [rng.permutation(n_tests) for _ in range(config.population_size)]

    fitnesses = np.array([fitness_fn(nest) for nest in nests])
    best_idx = int(np.argmax(fitnesses))
    best_ordering = nests[best_idx].copy()
    best_fitness = float(fitnesses[best_idx])
    history: list[float] = []

    n_abandon = max(1, int(round(config.pa * config.population_size)))

    for _ in range(config.max_iterations):
        # 1. Generate a new cuckoo via a Lévy flight around a random nest,
        #    biased toward the current global best (standard CSO move).
        for i in range(len(nests)):
            source = best_ordering if rng.random() < 0.5 else nests[i]
            candidate = _levy_perturb(source, config.levy_lambda, rng)
            candidate_fitness = fitness_fn(candidate)
            j = rng.integers(0, len(nests))
            if candidate_fitness > fitnesses[j]:
                nests[j] = candidate
                fitnesses[j] = candidate_fitness

        # 2. Abandon the worst pa fraction of nests and replace with fresh
        #    random permutations (global exploration).
        worst_idx = np.argsort(fitnesses)[:n_abandon]
        for idx in worst_idx:
            nests[idx] = rng.permutation(n_tests)
            fitnesses[idx] = fitness_fn(nests[idx])

        gen_best_idx = int(np.argmax(fitnesses))
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = float(fitnesses[gen_best_idx])
            best_ordering = nests[gen_best_idx].copy()
        history.append(best_fitness)

    return best_ordering, best_fitness, history
