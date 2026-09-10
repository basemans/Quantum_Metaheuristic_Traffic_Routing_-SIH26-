"""
traffic_simulation.py

Two congestion layers, both keeping congestion_factor >= 1.0:

1. Static base (existing): perturb every edge once before a run -
   congestion_factor = 1.0 + |N(0, variance_level)|. This feeds Phase 0's
   dynamic-k signal (candidate_set.py).

2. Flow feedback (added): given a ROUTE SET, count how many vehicles use
   each edge (link flows) and re-cost the network with a BPR-style volume-
   delay function (Bureau of Public Roads):
       true_cost(flow) = base_length * base_factor * (1 + alpha*(flow/capacity)^beta)
   This makes the collective problem non-decomposable: rerouting vehicles
   off crowded edges reduces OTHER vehicles' cost, so the globally optimal
   route set is not just "each vehicle's own shortest path" - this is what
   QPSO exploits (the greedy/identity assignment becomes a User-Equilibrium
   warm start, and QPSO searches toward the System Optimum).
"""

import random
import statistics
import networkx as nx


def apply_random_congestion(
    G: nx.DiGraph,
    variance_level: float = 0.2,
    seed: int = None
) -> None:
    """
    Perturbs every edge's congestion_factor in place.

    Parameters
    ----------
    variance_level : controls spread of congestion. 0.0 = no congestion
                      (all factors = 1.0, free-flow). Higher = more spread /
                      more volatile traffic. Currently implemented as the
                      standard deviation of a normal distribution around 1.0,
                      clipped to stay >= 1.0.
    """
    rng = random.Random(seed)

    for u, v in G.edges():
        factor = 1.0 + abs(rng.gauss(0, variance_level))
        G.edges[u, v]["congestion_factor"] = factor


def congestion_variance(G: nx.DiGraph) -> float:
    """Returns the variance of the STATIC base congestion_factor across all
    edges. This is the signal dynamic-k will read (see candidate_set.py)."""
    factors = [data["congestion_factor"] for _, _, data in G.edges(data=True)]
    if len(factors) < 2:
        return 0.0
    return statistics.variance(factors)


# ---------------------------------------------------------------------------
# Flow feedback (BPR volume-delay)
# ---------------------------------------------------------------------------
def route_flows(routes: dict) -> dict:
    """Aggregate per-edge vehicle counts over a route set.

    routes: {vehicle_id: {"path": [node, ...], ...}} -> {(u, v): count}"""
    flows = {}
    for r in routes.values():
        path = r["path"]
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            flows[edge] = flows.get(edge, 0) + 1
    return flows


def flow_load_multiplier(
    flow: float,
    capacity: float,
    alpha: float = 0.15,
    beta: float = 4.0
) -> float:
    """BPR volume-delay multiplier: >= 1.0, 1.0 at zero flow."""
    if capacity <= 0:
        raise ValueError("capacity must be > 0")
    if flow <= 0:
        return 1.0
    ratio = flow / capacity
    return 1.0 + alpha * ratio ** beta


def flow_edge_cost(
    G: nx.DiGraph,
    u, v,
    flow: float,
    capacity: float,
    alpha: float = 0.15,
    beta: float = 4.0
) -> float:
    """Flow-dependent true cost of edge (u, v):
    base_length * base_congestion * BPR(flow).

    Capacity resolution: if the edge carries a per-edge 'capacity' attribute
    (set by real_network_adapter for OSM roads, from lanes x road class), that
    physical value overrides the scalar `capacity` argument (which remains the
    fallback for synthetic grids).
    """
    data = G.edges[u, v]
    base_factor = data["congestion_factor"]
    edge_capacity = data.get("capacity", capacity)
    return data["base_length"] * base_factor * flow_load_multiplier(
        flow, edge_capacity, alpha, beta
    )


if __name__ == "__main__":
    from network_generator import create_grid_network

    G = create_grid_network(rows=4, cols=4, spacing=2000.0)

    print("Before congestion:")
    print("  variance =", congestion_variance(G))

    apply_random_congestion(G, variance_level=0.3, seed=42)

    print("After congestion (variance_level=0.3):")
    print("  variance =", congestion_variance(G))
    sample_edges = list(G.edges(data=True))[:5]
    for u, v, data in sample_edges:
        print(f"  {u}->{v}: congestion_factor={data['congestion_factor']:.3f}")
