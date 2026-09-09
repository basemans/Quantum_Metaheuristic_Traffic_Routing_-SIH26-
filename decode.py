"""
decode.py

Decode layer (methodology Section 5) + fitness (Section 6).

Given a particle's bias vector over candidate_edges, reweight the candidate
subgraph:
    effective_cost(u, v) = true_cost(u, v) * bias(u, v)
then route every vehicle over the candidate subgraph only.

Two solvers are provided, both implemented on top of heapq:
  - dijkstra : reference implementation (also used standalone for exact
               full-graph routing, e.g. the benchmarking baseline).
  - a_star   : same objective, A* heuristic = euclidean distance to the
               destination times the cheapest cost-per-meter found anywhere
               in the candidate subgraph (the "min possible speed" factor).
               That heuristic is admissible AND consistent, so A* must return
               exactly what Dijkstra returns - the two exist as a runtime
               comparison, not a quality one.

Fitness (Section 6) is scored on UNBIASED true_cost:
    fitness(particle) = sum over vehicles of path_true_cost(route)
The bias is a SEARCH HANDLE; true_cost is the OBJECTIVE.
"""

import heapq
import math

import networkx as nx

from network_generator import path_true_cost, true_cost
from traffic_simulation import flow_edge_cost, route_flows


# ---------------------------------------------------------------------------
# Reweighted costs
# ---------------------------------------------------------------------------
def effective_cost(G: nx.DiGraph, u, v, bias: dict = None) -> float:
    """true_cost(u, v) * bias(u, v). Missing edges in `bias` default to
    bias = 1.0 (so candidate edges untouched by the particle route as usual)."""
    if bias is None:
        bias = {}
    return true_cost(G, u, v) * bias.get((u, v), 1.0)


def _node_positions(G: nx.DiGraph) -> dict:
    return {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G}


def _min_cost_per_meter(G: nx.DiGraph, allowed_edges: set, bias: dict) -> float:
    """Cheapest effective cost per meter across the candidate subgraph.
    Scales the A* heuristic ("min possible speed") so the heuristic stays
    admissible: any route to the destination is at least
    (euclidean distance) * (this ratio)."""
    best = None
    for u, v in allowed_edges:
        ratio = effective_cost(G, u, v, bias) / G.edges[u, v]["base_length"]
        if best is None or ratio < best:
            best = ratio
    if best is None:
        raise ValueError("allowed_edges is empty - cannot compute A* heuristic scale")
    return best


def _reconstruct(prev: dict, start, target) -> list:
    path = []
    node = target
    while node != start:
        path.append(node)
        node = prev[node]
    path.append(start)
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Solvers (candidate-subgraph restricted)
# ---------------------------------------------------------------------------
def dijkstra(
    G: nx.DiGraph,
    start,
    target,
    bias: dict = None,
    allowed_edges: set = None,
):
    """Dijkstra over effective_cost, restricted to allowed_edges.
    allowed_edges=None -> full graph. Returns (path, effective_cost)."""
    if bias is None:
        bias = {}
    if allowed_edges is None:
        allowed_edges = set(G.edges())

    dist = {start: 0.0}
    prev = {}
    pq = [(0.0, start)]
    closed = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in closed:
            continue
        closed.add(u)
        if u == target:
            break
        for v in G.successors(u):
            if (u, v) not in allowed_edges:
                continue
            nd = d + effective_cost(G, u, v, bias)
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if target not in dist:
        raise ValueError(f"No path from {start} to {target} in the allowed subgraph")
    return _reconstruct(prev, start, target), dist[target]


def a_star(
    G: nx.DiGraph,
    start,
    target,
    bias: dict = None,
    allowed_edges: set = None,
):
    """A* over effective_cost with the admissible/consistent euclidean
    heuristic described in the module docstring. Same contract as dijkstra."""
    if bias is None:
        bias = {}
    if allowed_edges is None:
        allowed_edges = set(G.edges())

    pos = _node_positions(G)
    scale = _min_cost_per_meter(G, allowed_edges, bias)
    tx, ty = pos[target]

    def heuristic(n) -> float:
        if n == target:
            return 0.0
        nx_, ny_ = pos[n]
        return math.hypot(tx - nx_, ty - ny_) * scale

    g = {start: 0.0}
    prev = {}
    open_heap = [(heuristic(start), start)]
    closed = set()

    while open_heap:
        f, u = heapq.heappop(open_heap)
        if u in closed:
            continue
        closed.add(u)
        if u == target:
            break
        for v in G.successors(u):
            if (u, v) not in allowed_edges:
                continue
            ng = g[u] + effective_cost(G, u, v, bias)
            if ng < g.get(v, float("inf")):
                g[v] = ng
                prev[v] = u
                heapq.heappush(open_heap, (ng + heuristic(v), v))

    if target not in g:
        raise ValueError(f"No path from {start} to {target} in the allowed subgraph")
    return _reconstruct(prev, start, target), g[target]


# ---------------------------------------------------------------------------
# Per-particle decode + fitness
# ---------------------------------------------------------------------------
def route_all_vehicles(
    G: nx.DiGraph,
    vehicles: list,
    bias: dict = None,
    allowed_edges: set = None,
    algorithm: str = "dijkstra",
) -> dict:
    """Route every vehicle over the (candidate) subgraph under `bias`.

    Returns {vehicle_id: {"path": [...], "effective_cost": float,
                          "true_cost": float}}.
    """
    if bias is None:
        bias = {}
    if algorithm not in ("dijkstra", "astar"):
        raise ValueError(f"algorithm must be 'dijkstra' or 'astar', got {algorithm!r}")

    solver = a_star if algorithm == "astar" else dijkstra
    routes = {}
    for vehicle in vehicles:
        path, eff_cost = solver(
            G,
            vehicle["start"],
            vehicle["destination"],
            bias=bias,
            allowed_edges=allowed_edges,
        )
        routes[vehicle["id"]] = {
            "path": path,
            "effective_cost": eff_cost,
            "true_cost": path_true_cost(G, path),
        }
    return routes


def fitness(G: nx.DiGraph, routes: dict) -> float:
    """Section 6: sum of UNBIASED true_cost over all routed vehicles."""
    return sum(r["true_cost"] for r in routes.values())


def fitness_with_feedback(
    G: nx.DiGraph,
    routes: dict,
    capacity: float,
    alpha: float = 0.15,
    beta: float = 4.0
) -> float:
    """System-total travel time under FLOW-DEPENDENT (BPR) congestion.

    Counts each edge's vehicle flow from the route set, re-costs every edge
    with the BPR volume-delay multiplier (traffic_simulation.py), then sums
    the joint route costs. Because edge cost now depends on how MANY vehicles
    share it, the fitness is non-decomposable: the sum of individually-
    shortest routes (greedy identity flow) is generally NOT optimal, which is
    exactly the coordination problem QPSO is meant to win.
    """
    flows = route_flows(routes)
    total = 0.0
    for r in routes.values():
        path = r["path"]
        for i in range(len(path) - 1):
            u, w = path[i], path[i + 1]
            total += flow_edge_cost(G, u, w, flows[(u, w)], capacity, alpha, beta)
    return total


def evaluate(
    G: nx.DiGraph,
    vehicles: list,
    bias: dict,
    allowed_edges: set,
    algorithm: str = "dijkstra",
):
    """Convenience: route + score in one call. Returns (routes, fitness)."""
    routes = route_all_vehicles(
        G, vehicles, bias=bias, allowed_edges=allowed_edges, algorithm=algorithm
    )
    return routes, fitness(G, routes)


def score_function(
    G: nx.DiGraph,
    scoring: str = "static",
    capacity: float = None,
    alpha: float = 0.15,
    beta: float = 4.0,
):
    """Returns a route-set -> fitness callable shared by ALL optimizers
    (QPSO, classical PSO, GA) so the benchmark is apples-to-apples.

    scoring="static": fitness(G, routes)  (decomposable; Section 6)
    scoring="flow"  : fitness_with_feedback (BPR; non-decomposable)
    """
    if scoring not in ("static", "flow"):
        raise ValueError(f"scoring must be 'static' or 'flow', got {scoring!r}")
    if scoring == "static":
        return lambda routes: fitness(G, routes)
    if capacity is None:
        raise ValueError("scoring='flow' requires a positive `capacity`")
    return lambda routes: fitness_with_feedback(G, routes, capacity, alpha=alpha, beta=beta)


if __name__ == "__main__":
    from network_generator import create_grid_network
    from vehicle_generator import create_vehicles
    from traffic_simulation import apply_random_congestion
    from candidate_set import build_candidate_edge_set

    G = create_grid_network(rows=6, cols=6, spacing=2000.0)
    apply_random_congestion(G, variance_level=0.3, seed=42)
    vehicles = create_vehicles(G, num_vehicles=10, hub_ratio=1.5, seed=1)
    result = build_candidate_edge_set(G, vehicles, verbose=False)

    bias = {}  # identity bias: candidate-subgraph decode == shortest paths
    routes, fit = evaluate(
        G, vehicles, bias, result["candidate_edges"], algorithm="dijkstra"
    )
    routes_astar, _ = evaluate(
        G, vehicles, bias, result["candidate_edges"], algorithm="astar"
    )

    print(f"Candidate edges: {len(result['candidate_edges'])} "
          f"(reduction_ratio={result['reduction_ratio']:.3f})")
    print(f"Fitness (identity bias, Dijkstra): {fit:.2f}")
    print(f"Fitness (identity bias, A*):       {fitness(G, routes_astar):.2f}")
    print(f"(A* == Dijkstra is expected: {fitness(G, routes_astar) == fit})")