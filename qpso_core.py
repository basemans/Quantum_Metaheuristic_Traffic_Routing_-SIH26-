"""
qpso_core.py

Quantum-Inspired Particle Swarm Optimization over the candidate-edge bias
vector (methodology Section 4). Single-threaded for now (Section 7
parallelization via multiprocessing.Pool is a later step).

Particle position = bias vector over candidate_edges only:
    x_i = { bias(u, v) : (u, v) in candidate_edges }

Per dimension, per particle, per iteration (fresh draws each time):
    phi   ~ U(0, 1)
    p     = phi * pbest_i + (1 - phi) * gbest
    u     ~ U(0, 1)
    sign  in {+1, -1}  (random)
    x_new = p + sign * beta * |x - mbest| * ln(1/u)
    x_new <- clip(x_new, bias_min, bias_max)

  beta = contraction-expansion coefficient, annealed over iterations
         (default 1.0 -> 0.5 linear).
  mbest = elementwise mean of pbest across the swarm.

No velocity term - this fully replaces PSO's w, c1, c2 update.

Fitness = sum over vehicles of UNBIASED true_cost (decode.py, Section 6).
Minimization. The bias is a search handle only.

Design note: particle 0 is seeded at identity bias (all 1.0). Its first
decoded fitness therefore equals plain shortest-path routing, so the global
best is provably never worse than the identity-bias baseline.
"""

import math
import random
import time

import networkx as nx

from decode import fitness, fitness_with_feedback, route_all_vehicles


def _clip(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else (hi if value > hi else value)


def _swarm_mean_best(pbest_list: list, edges: list) -> dict:
    """Elementwise mean of pbest vectors across the swarm (mbest)."""
    n = len(pbest_list)
    mbest = {}
    for edge in edges:
        mbest[edge] = sum(p[edge] for p in pbest_list) / n
    return mbest


def _annealed_beta(t: int, iterations: int, beta_start: float, beta_end: float) -> float:
    """Linear contraction-expansion schedule: 1.0 -> 0.5 over the run."""
    if iterations <= 1:
        return beta_end
    return beta_start + (beta_end - beta_start) * (t / (iterations - 1))


def run_qpso(
    G: nx.DiGraph,
    vehicles: list,
    candidate_edges: set,
    num_particles: int = 20,
    iterations: int = 30,
    beta_start: float = 1.0,
    beta_end: float = 0.5,
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
    """
    Run the QPSO optimizer. Minimizes fitness over the bias vector.

    Parameters
    ----------
    scoring : "static"  -> sum of unbiased true_cost on the static graph
                          (Section 6, decomposable - identity bias IS optimal)
              "flow"    -> system-total travel time under BPR flow feedback
                          (non-decomposable - QPSO can beat greedy shortest
                          paths by coordinating the route set)
    capacity, alpha, beta : BPR volume-delay parameters for scoring="flow".

    Returns
    -------
    dict with:
        gbest           : best bias vector found (edge -> bias)
        gbest_fitness   : its fitness under the chosen scoring
        gbest_routes    : route dict decoded from gbest
        history         : [best fitness per iteration]  (monotone non-increasing)
        mean_history    : [mean swarm fitness per iteration]
        elapsed_sec     : wall-clock time
    """
    if num_particles < 1:
        raise ValueError("num_particles must be >= 1")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if not candidate_edges:
        raise ValueError("candidate_edges is empty - nothing to optimize over")
    if bias_min > 1.0 or bias_max < 1.0:
        # identity bias (1.0) must stay reachable so the baseline guarantee holds
        raise ValueError("bias range must include 1.0 (bias_min <= 1.0 <= bias_max)")
    if scoring not in ("static", "flow"):
        raise ValueError(f"scoring must be 'static' or 'flow', got {scoring!r}")

    # decode routes, then score under the chosen objective
    def score(routes: dict) -> float:
        if scoring == "static":
            return fitness(G, routes)
        return fitness_with_feedback(G, routes, capacity, alpha=alpha, beta=beta)

    edges = sorted(candidate_edges)
    rng = random.Random(seed)

    # ---- initialization ------------------------------------------------
    positions = []
    pbests = []
    pbest_fitness = []

    for i in range(num_particles):
        if i == 0:
            # identity seeding: baseline = plain shortest-path routing
            pos = {edge: 1.0 for edge in edges}
        else:
            pos = {edge: rng.uniform(bias_min, bias_max) for edge in edges}
        positions.append(pos)

        routes = route_all_vehicles(
            G, vehicles, pos, candidate_edges, algorithm=algorithm
        )
        fit = score(routes)
        pbests.append(pos)
        pbest_fitness.append(fit)

    best_idx = min(range(num_particles), key=lambda i: pbest_fitness[i])
    gbest = dict(pbests[best_idx])
    gbest_fitness = pbest_fitness[best_idx]

    history = [gbest_fitness]
    mean_history = [sum(pbest_fitness) / num_particles]

    start = time.perf_counter()

    # ---- iterations -----------------------------------------------------
    for t in range(iterations):
        beta = _annealed_beta(t, iterations, beta_start, beta_end)
        mbest = _swarm_mean_best(pbests, edges)

        for i in range(num_particles):
            pos = positions[i]
            new_pos = {}

            for edge in edges:
                phi = rng.random()
                p = phi * pbests[i][edge] + (1.0 - phi) * gbest[edge]

                u = rng.random()
                if u <= 0.0:
                    u = 1e-12  # guard ln(0) = -inf
                dist = abs(pos[edge] - mbest[edge])
                sign = 1.0 if rng.random() < 0.5 else -1.0
                step = sign * beta * dist * (-math.log(u))

                new_pos[edge] = _clip(p + step, bias_min, bias_max)

            positions[i] = new_pos

            routes = route_all_vehicles(
                G, vehicles, new_pos, candidate_edges, algorithm=algorithm
            )
            fit = score(routes)

            if fit < pbest_fitness[i]:
                pbests[i] = new_pos
                pbest_fitness[i] = fit

        best_idx = min(range(num_particles), key=lambda i: pbest_fitness[i])
        if pbest_fitness[best_idx] < gbest_fitness:
            gbest = dict(pbests[best_idx])
            gbest_fitness = pbest_fitness[best_idx]

        history.append(gbest_fitness)
        mean_history.append(sum(pbest_fitness) / num_particles)

        if verbose:
            print(
                f"iter {t + 1:>3}/{iterations}: best={gbest_fitness:.2f} "
                f"mean={mean_history[-1]:.2f} beta={beta:.3f}"
            )

    elapsed = time.perf_counter() - start
    gbest_routes = route_all_vehicles(
        G, vehicles, gbest, candidate_edges, algorithm=algorithm
    )

    return {
        "gbest": gbest,
        "gbest_fitness": gbest_fitness,
        "gbest_routes": gbest_routes,
        "history": history,
        "mean_history": mean_history,
        "elapsed_sec": elapsed,
    }


if __name__ == "__main__":
    from network_generator import create_grid_network
    from vehicle_generator import create_vehicles
    from traffic_simulation import apply_random_congestion
    from candidate_set import build_candidate_edge_set
    from decode import evaluate, fitness, fitness_with_feedback, route_all_vehicles

    G = create_grid_network(rows=6, cols=6, spacing=2000.0)
    apply_random_congestion(G, variance_level=0.3, seed=42)
    vehicles = create_vehicles(G, num_vehicles=10, hub_ratio=1.5, seed=1)
    result = build_candidate_edge_set(G, vehicles, verbose=False)

    routes_base, fit_base = evaluate(G, vehicles, {}, result["candidate_edges"])
    print(f"Identity-bias baseline fitness (static): {fit_base:.2f}")

    swarm = run_qpso(
        G, vehicles, result["candidate_edges"],
        num_particles=20, iterations=30, seed=42,
    )
    print(f"QPSO best fitness (static):            {swarm['gbest_fitness']:.2f}")
    print(f"Static baseline beaten/equaled:        {swarm['gbest_fitness'] <= fit_base}")

    print()
    print("=== Flow-feedback scoring (BPR congestion, non-decomposable) ===")
    cap = len(vehicles) / 3.0
    fit_ue = fitness_with_feedback(G, routes_base, capacity=cap)
    print(f"Greedy shortest-path flow fitness (User-Equilibrium): {fit_ue:.2f}")
    swarm_flow = run_qpso(
        G, vehicles, result["candidate_edges"],
        num_particles=25, iterations=40, seed=42,
        scoring="flow", capacity=cap, verbose=True,
    )
    print(f"\nQPSO best fitness (flow):               {swarm_flow['gbest_fitness']:.2f}")
    print(f"Flow baseline beaten:                   {swarm_flow['gbest_fitness'] < fit_ue}")
    print(f"Improvement: {(1 - swarm_flow['gbest_fitness'] / fit_ue) * 100:.2f}%")
    print(f"Elapsed: {swarm_flow['elapsed_sec']:.3f}s")