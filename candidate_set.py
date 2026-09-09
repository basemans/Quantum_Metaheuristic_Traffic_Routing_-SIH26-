"""
candidate_set.py

Phase 0 of the pipeline: precompute, once per run, a reduced "candidate
edge set" that the QPSO bias vector will operate over (instead of every
edge in the whole graph).

For each vehicle:
    1. Determine k dynamically from current congestion variance (Open
       Question #1 - mapping function below is a placeholder, tunable).
    2. Compute that vehicle's k-shortest-paths (Yen's algorithm, via
       networkx's shortest_simple_paths) using true_cost as weight.
    3. Collect every edge appearing in any of those paths.

Union across all vehicles -> candidate_edges. This defines:
    - the dimensionality of the QPSO bias vector (qpso_core.py)
    - the subgraph that Dijkstra/A* search during decode (decode.py)

Also reports reduction_ratio = |candidate_edges| / |E(G)|, which is the
metric that empirically tests the spatial-locality assumption (random vs.
clustered vehicle demand).
"""

import networkx as nx
from network_generator import true_cost


# --- Dynamic k mapping (Open Question #1 - placeholder, tune later) ---
# Linear interpolation between (k_min at variance=0) and (k_max at
# variance=variance_at_kmax), clipped to [k_min, k_max].
def dynamic_k(
    congestion_var: float,
    k_min: int = 5,
    k_max: int = 15,
    variance_at_kmax: float = 0.1
) -> int:
    """Maps current congestion variance to a k value for Yen's algorithm.
    Higher variance (more volatile/uneven congestion) -> larger k, so more
    alternate routes are available for QPSO to consider."""
    if congestion_var <= 0:
        return k_min
    ratio = min(congestion_var / variance_at_kmax, 1.0)
    k = k_min + ratio * (k_max - k_min)
    return int(round(k))


def vehicle_k_shortest_paths(G: nx.DiGraph, start, destination, k: int) -> list:
    """Returns up to k shortest paths (as node-lists) from start to
    destination, ranked by true_cost. Uses networkx's generator-based Yen's
    algorithm implementation (shortest_simple_paths)."""
    try:
        path_generator = nx.shortest_simple_paths(
            G, start, destination,
            weight=lambda u, v, d: d["base_length"] * d["congestion_factor"]
        )
        paths = []
        for i, path in enumerate(path_generator):
            if i >= k:
                break
            paths.append(path)
        return paths
    except nx.NetworkXNoPath:
        return []


def edges_in_path(path: list) -> set:
    return set(zip(path[:-1], path[1:]))


def build_candidate_edge_set(
    G: nx.DiGraph,
    vehicles: list,
    k_min: int = 5,
    k_max: int = 15,
    variance_at_kmax: float = 0.1,
    verbose: bool = True
) -> dict:
    """
    Runs Phase 0 for a full vehicle list.

    Returns
    -------
    dict with:
        candidate_edges : set of (u, v) tuples
        k_used          : the dynamic k value used for this run
        per_vehicle_paths : {vehicle_id: [path, path, ...]}  (kept for
                             potential later use / debugging)
        reduction_ratio : |candidate_edges| / |E(G)|
    """
    from traffic_simulation import congestion_variance

    var = congestion_variance(G)
    k = dynamic_k(var, k_min=k_min, k_max=k_max, variance_at_kmax=variance_at_kmax)

    candidate_edges = set()
    per_vehicle_paths = {}
    unreachable = []

    for vehicle in vehicles:
        paths = vehicle_k_shortest_paths(G, vehicle["start"], vehicle["destination"], k)
        per_vehicle_paths[vehicle["id"]] = paths

        if not paths:
            unreachable.append(vehicle["id"])
            continue

        for path in paths:
            candidate_edges.update(edges_in_path(path))

    total_edges = G.number_of_edges()
    reduction_ratio = len(candidate_edges) / total_edges if total_edges else 0.0

    if verbose:
        print(f"Congestion variance: {var:.5f}  ->  dynamic k = {k}")
        print(f"Candidate edges: {len(candidate_edges)} / {total_edges} "
              f"total  (reduction_ratio = {reduction_ratio:.3f})")
        if unreachable:
            print(f"WARNING: {len(unreachable)} vehicle(s) have no path: {unreachable}")

    return {
        "candidate_edges": candidate_edges,
        "k_used": k,
        "per_vehicle_paths": per_vehicle_paths,
        "reduction_ratio": reduction_ratio,
    }


if __name__ == "__main__":
    from network_generator import create_grid_network
    from vehicle_generator import create_vehicles
    from traffic_simulation import apply_random_congestion

    G = create_grid_network(rows=6, cols=6, spacing=2000.0)
    apply_random_congestion(G, variance_level=0.3, seed=42)

    print("=" * 60)
    print("Fully random demand (hub_ratio=0, worst case for reduction)")
    print("=" * 60)
    random_vehicles = create_vehicles(G, num_vehicles=10, hub_ratio=0, seed=1)
    result_random = build_candidate_edge_set(G, random_vehicles)

    print()
    print("=" * 60)
    print("Mostly hub-based demand (hub_ratio=9, best case for reduction)")
    print("=" * 60)
    clustered_vehicles = create_vehicles(
        G, num_vehicles=10, hub_ratio=9, num_hubs=2, hub_radius=1.0, seed=1
    )
    result_clustered = build_candidate_edge_set(G, clustered_vehicles)

    print()
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Random    reduction_ratio: {result_random['reduction_ratio']:.3f}")
    print(f"Clustered reduction_ratio: {result_clustered['reduction_ratio']:.3f}")
