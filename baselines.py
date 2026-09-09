"""
baselines.py

Classical metaheuristics for the benchmarking harness (methodology Section 8).
All optimize the SAME objective as QPSO (the bias-vector / route-set problem
via decode.py's score_function), so the benchmark is apples-to-apples:

  run_pso : classical PSO with w / c1 / c2 velocity update over the bias
            vector (the "discrete"/continuous counterpart of QPSO - it keeps
            a velocity term; QPSO deliberately does not).
  run_ga  : real-coded genetic algorithm over the bias vector (BLX-alpha
            crossover, Gaussian mutation, tournament selection, elitism).

(An Ant Colony System baseline was implemented but REMOVED after benchmarking:
its route-construction search cannot discover coordinated flow-diversions -- it
recovered ~0% on every default-sweep config while bias-space methods reached
0.4-5%; see docs 4.10.)

Fairness conventions (identical to run_qpso):
  - population/particle count and iteration count are passed in
  - particle 0 / individual 0 = IDENTITY routing (plain shortest paths), so
    every method is seeded with the greedy User-Equilibrium solution and can
    provably never report worse than it
  - all results share run_qpso's return shape: gbest, gbest_fitness,
    gbest_routes, history (monotone), mean_history, elapsed_sec
  - deterministic via `seed`
"""

import random
import time

import networkx as nx

from decode import route_all_vehicles, score_function


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _identity_vector(edges: list) -> list:
    return [1.0] * len(edges)


def _random_vector(edges: list, bias_min: float, bias_max: float, rng: random.Random) -> list:
    return [rng.uniform(bias_min, bias_max) for _ in edges]


def _clip(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else (hi if value > hi else value)


def _decode_and_score(
    G, vehicles, candidate_edges, genome, edges, score, algorithm
):
    """genome (list over `edges`) -> routes + fitness."""
    pos = dict(zip(edges, genome))
    routes = route_all_vehicles(G, vehicles, pos, candidate_edges, algorithm=algorithm)
    return routes, score(routes)


def _pack_result(gbest_genome, gbest_fitness, gbest_routes, history,
                 mean_history, elapsed, edges) -> dict:
    return {
        "gbest": dict(zip(edges, gbest_genome)),
        "gbest_fitness": gbest_fitness,
        "gbest_routes": gbest_routes,
        "history": history,
        "mean_history": mean_history,
        "elapsed_sec": elapsed,
    }


# ---------------------------------------------------------------------------
# Classical PSO (continuous bias space, w/c1/c2)
# ---------------------------------------------------------------------------
def run_pso(
    G: nx.DiGraph,
    vehicles: list,
    candidate_edges: set,
    num_particles: int = 20,
    iterations: int = 30,
    c1: float = 2.0,
    c2: float = 2.0,
    inertia_start: float = 0.9,
    inertia_end: float = 0.4,
    bias_min: float = 0.5,
    bias_max: float = 1.5,
    seed: int = None,
    algorithm: str = "dijkstra",
    scoring: str = "static",
    capacity: float = 8.0,
    alpha: float = 0.15,
    beta: float = 4.0,
    verbose: bool = False,
) -> dict:
    if num_particles < 1 or iterations < 1:
        raise ValueError("num_particles and iterations must be >= 1")
    edges = sorted(candidate_edges)
    d = len(edges)
    rng = random.Random(seed)
    score = score_function(G, scoring=scoring, capacity=capacity, alpha=alpha, beta=beta)

    positions = []
    velocities = []
    pbests = []
    pbest_fitness = []

    for i in range(num_particles):
        pos = _identity_vector(edges) if i == 0 else _random_vector(edges, bias_min, bias_max, rng)
        positions.append(pos)
        velocities.append([0.0] * d)
        routes, fit = _decode_and_score(G, vehicles, candidate_edges, pos, edges, score, algorithm)
        pbests.append(list(pos))
        pbest_fitness.append(fit)

    best_idx = min(range(num_particles), key=lambda i: pbest_fitness[i])
    gbest = list(pbests[best_idx])
    gbest_fitness = pbest_fitness[best_idx]

    history = [gbest_fitness]
    mean_history = [sum(pbest_fitness) / num_particles]
    start = time.perf_counter()

    for t in range(iterations):
        w = inertia_start + (inertia_end - inertia_start) * (t / max(iterations - 1, 1))

        for i in range(num_particles):
            for j in range(d):
                r1, r2 = rng.random(), rng.random()
                velocities[i][j] = (
                    w * velocities[i][j]
                    + c1 * r1 * (pbests[i][j] - positions[i][j])
                    + c2 * r2 * (gbest[j] - positions[i][j])
                )
                positions[i][j] = _clip(positions[i][j] + velocities[i][j], bias_min, bias_max)

            routes, fit = _decode_and_score(
                G, vehicles, candidate_edges, positions[i], edges, score, algorithm
            )
            if fit < pbest_fitness[i]:
                pbests[i] = list(positions[i])
                pbest_fitness[i] = fit

        best_idx = min(range(num_particles), key=lambda i: pbest_fitness[i])
        if pbest_fitness[best_idx] < gbest_fitness:
            gbest = list(pbests[best_idx])
            gbest_fitness = pbest_fitness[best_idx]

        history.append(gbest_fitness)
        mean_history.append(sum(pbest_fitness) / num_particles)

        if verbose:
            print(f"pso iter {t + 1:>3}: best={gbest_fitness:.2f}")

    elapsed = time.perf_counter() - start
    gbest_routes = route_all_vehicles(G, vehicles, dict(zip(edges, gbest)), candidate_edges)
    return _pack_result(gbest, gbest_fitness, gbest_routes, history, mean_history, elapsed, edges)


# ---------------------------------------------------------------------------
# Genetic algorithm (real-coded)
# ---------------------------------------------------------------------------
def run_ga(
    G: nx.DiGraph,
    vehicles: list,
    candidate_edges: set,
    pop_size: int = 20,
    generations: int = 30,
    crossover_prob: float = 0.8,
    mutation_prob: float = 0.1,
    blx_alpha: float = 0.5,
    tournament_size: int = 3,
    elite: int = 1,
    bias_min: float = 0.5,
    bias_max: float = 1.5,
    seed: int = None,
    algorithm: str = "dijkstra",
    scoring: str = "static",
    capacity: float = 8.0,
    alpha: float = 0.15,
    beta: float = 4.0,
    verbose: bool = False,
) -> dict:
    if pop_size < 2 or generations < 1:
        raise ValueError("pop_size >= 2 and generations >= 1")
    edges = sorted(candidate_edges)
    d = len(edges)
    rng = random.Random(seed)
    score = score_function(G, scoring=scoring, capacity=capacity, alpha=alpha, beta=beta)
    sigma = (bias_max - bias_min) / 10.0

    population = [_identity_vector(edges)]
    while len(population) < pop_size:
        population.append(_random_vector(edges, bias_min, bias_max, rng))

    fits = []
    for ind in population:
        _, fit = _decode_and_score(G, vehicles, candidate_edges, ind, edges, score, algorithm)
        fits.append(fit)

    best_idx = min(range(pop_size), key=lambda i: fits[i])
    gbest = list(population[best_idx])
    gbest_fitness = fits[best_idx]

    history = [gbest_fitness]
    mean_history = [sum(fits) / pop_size]
    start = time.perf_counter()

    for gen in range(generations):
        order = sorted(range(pop_size), key=lambda i: fits[i])[:elite]
        new_pop = [list(population[i]) for i in order]

        def tournament():
            idxs = rng.sample(range(pop_size), min(tournament_size, pop_size))
            return min(idxs, key=lambda i: fits[i])

        while len(new_pop) < pop_size:
            p1 = list(population[tournament()])
            p2 = list(population[tournament()])

            if rng.random() < crossover_prob:
                for j in range(d):
                    lo, hi = min(p1[j], p2[j]), max(p1[j], p2[j])
                    span = hi - lo
                    gamma = rng.uniform(-blx_alpha, 1.0 + blx_alpha)
                    p1[j] = _clip(lo + gamma * span, bias_min, bias_max)
                child = p1
            else:
                child = [p1[j] if rng.random() < 0.5 else p2[j] for j in range(d)]

            for j in range(d):
                if rng.random() < mutation_prob:
                    child[j] = _clip(child[j] + rng.gauss(0, sigma), bias_min, bias_max)

            new_pop.append(child)

        population = new_pop
        fits = []
        for ind in population:
            _, fit = _decode_and_score(G, vehicles, candidate_edges, ind, edges, score, algorithm)
            fits.append(fit)

        best_idx = min(range(pop_size), key=lambda i: fits[i])
        if fits[best_idx] < gbest_fitness:
            gbest = list(population[best_idx])
            gbest_fitness = fits[best_idx]

        history.append(gbest_fitness)
        mean_history.append(sum(fits) / pop_size)

        if verbose:
            print(f"ga  gen {gen + 1:>3}: best={gbest_fitness:.2f}")

    elapsed = time.perf_counter() - start
    gbest_routes = route_all_vehicles(G, vehicles, dict(zip(edges, gbest)), candidate_edges)
    return _pack_result(gbest, gbest_fitness, gbest_routes, history, mean_history, elapsed, edges)