"""
real_network_adapter.py

Ingests a small-scale REAL road network from OpenStreetMap (via osmnx) and
converts it into the project's native DiGraph format so the entire existing
pipeline (Phase 0 -> decode -> QPSO) runs unchanged.

Conversion contract (matches network_generator.create_grid_network output):
    nodes   : every node carries {'x': meters, 'y': meters} in a LOCAL
              projected CRS (osmnx projects the graph to a UTM zone, so edge
              lengths and coordinates are true meters - no degrees).
    edges   : every directed edge carries
                  base_length       : meters (from OSM 'length'), strictly > 0
                  congestion_factor : 1.0 (free flow; perturbed later by
                                      traffic_simulation.apply_random_congestion)

Choices that keep the pipeline sound:
    - network_type='drive'         : drivable roads only (our vehicles route)
    - graph_from_point(dist=...)   : small, controllable geographic footprint
    - simplify=True                : contract OSM geometry into fewer
                                     intersection nodes (fewer, longer edges)
    - one edge per direction       : parallel/duplicate OSM edges are merged
                                     to the shortest length
    - zero-length edges dropped    : true_cost / base_length (the A* heuristic
                                     scale, decode._min_cost_per_meter) would
                                     divide by zero otherwise
    - largest weakly-connected     : keeps the demo free of the "no path" case
      component only

Demo (`__main__`): downloads a ~2 km district around a configurable center,
then runs the exact same pipeline as qpso_core's `__main__` demo (static
identity baseline, then QPSO under BPR flow feedback) and reports
reduction_ratio + improvement vs the greedy User-Equilibrium baseline.
"""

import networkx as nx

import osmnx as ox


def _to_simple_digraph(Gp: nx.MultiDiGraph) -> nx.DiGraph:
    """Converts a projected osmnx MultiDiGraph into the project's DiGraph
    contract (one directed edge per direction, base_length > 0). Keeps the
    OSM road metadata needed later for physical capacities (highway class,
    lane count, oneway flag)."""
    G = nx.DiGraph()
    for node, data in Gp.nodes(data=True):
        G.add_node(node, x=data["x"], y=data["y"])

    for u, v, k, data in Gp.edges(keys=True, data=True):
        length = data.get("length")
        if length is None or length <= 0.0:
            continue
        if not G.has_edge(u, v) or G.edges[u, v]["base_length"] > length:
            highway = data.get("highway")
            tags = {
                "base_length": float(length),
                "congestion_factor": 1.0,
                "highway": highway if isinstance(highway, str) else "residential",
                "lanes": int(data["lanes"]) if data.get("lanes") is not None else 1,
                "oneway": str(data.get("oneway", "")).lower()
                in ("yes", "true", "1"),
            }
            G.add_edge(u, v, **tags)
    return G


# Peak-hour per-lane capacity (vehicles/hour/lane) by road class - a coarse
# but physical stand-in for slow-move/freeway differences (HCM-style values).
_CAPACITY_PER_LANE = {
    "motorway": 2000, "motorway_link": 1500, "trunk": 1800, "trunk_link": 1500,
    "primary": 1600, "primary_link": 1400, "secondary": 1300, "secondary_link": 1200,
    "tertiary": 1200, "tertiary_link": 1100, "residential": 600,
    "living_street": 300, "unclassified": 800, "service": 400,
}

def attach_physical_capacity(G: nx.DiGraph) -> None:
    """Adds a per-edge 'capacity' (vehicles/hour) from OSM metadata:
    lanes x per-lane class capacity, halved for two-way streets (each
    direction handles about half the lanes). Note: hours, while our BPR
    flows are instantaneous counts - the ratio f/c is what matters, and it
    no longer scales away with N like the N/3 heuristic did."""
    for u, v, data in G.edges(data=True):
        per_lane = _CAPACITY_PER_LANE.get(data.get("highway", "residential"), 1000)
        lanes = max(1, data.get("lanes", 1))
        if data.get("oneway"):
            G.edges[u, v]["capacity"] = float(lanes * per_lane)
        else:
            G.edges[u, v]["capacity"] = float(lanes * per_lane / 2.0)


def _largest_weak_component(G: nx.DiGraph) -> nx.DiGraph:
    """Keeps only the largest weakly-connected component so every node can
    reach every other via the undirected skeleton (avoids unreachable O-D
    pairs on real maps, which are fragmented)."""
    comp = max(nx.weakly_connected_components(G), key=len)
    return G.subgraph(comp).copy()


def load_osm_network(
    center: tuple = None,
    dist: float = 1200.0,
    network_type: str = "drive",
    simplify: bool = True,
    physical_capacity: bool = False,
) -> dict:
    """
    Downloads a real road network around `center` (lat, lon) and converts it.

    center             : default (26.9124, 75.7873) - Jaipur city centre.
    dist               : radius in meters; the footprint is a square of side ~2*dist.
    physical_capacity  : attach per-edge 'capacity' from OSM lanes/road class
                         (traffic_simulation.flow_edge_cost then uses it
                         instead of the global N/3 fallback).
    """
    if center is None:
        center = (26.9124, 75.7873)

    ox.settings.use_cache = True
    ox.settings.overpass_settings = '[out:json][timeout:180]'

    Graw = ox.graph_from_point(
        center_point=center,
        dist=dist,
        network_type=network_type,
        simplify=simplify,
        retain_all=True,
    )
    if len(Graw) == 0:
        raise ValueError("OSM download returned an empty graph - check centre/network")

    Gp = ox.projection.project_graph(Graw, to_crs=None)
    G = _to_simple_digraph(Gp)
    G = _largest_weak_component(G)

    if physical_capacity:
        attach_physical_capacity(G)

    return {
        "G": G,
        "center": center,
        "dist_m": dist,
        "raw_nodes": len(Graw),
        "raw_edges": Graw.number_of_edges(),
        "nodes": len(G),
        "edges": G.number_of_edges(),
        "crs": Gp.graph.get("crs", "unknown"),
    }


if __name__ == "__main__":
    import argparse

    from network_generator import path_true_cost
    from vehicle_generator import create_vehicles
    from traffic_simulation import apply_random_congestion, congestion_variance
    from candidate_set import build_candidate_edge_set
    from decode import evaluate, fitness, fitness_with_feedback, route_all_vehicles
    from qpso_core import run_qpso

    parser = argparse.ArgumentParser(description="Real-road pipeline demo (OSM)")
    parser.add_argument("--center", nargs=2, type=float, default=None,
                        help="lat lon of district center (default Jaipur centre)")
    parser.add_argument("--dist", type=float, default=1000.0, help="radius (m)")
    parser.add_argument("--vehicles", type=int, default=10,
                        help="number of vehicles (default 10)")
    parser.add_argument("--od", choices=["synthetic", "realistic"], default="synthetic",
                        help="synthetic = blended hub/random pairs; realistic = "
                             "residential -> school/college/office demands from "
                             "real OSM land-use/building features (Jaipur-specific)")
    parser.add_argument("--physical-capacity", action="store_true",
                        help="use per-edge OSM road capacities instead of the N/3 "
                             "scaling heuristic for BPR congestion")
    parser.add_argument("--pop", type=int, default=25, help="QPSO particles")
    parser.add_argument("--iters", type=int, default=40, help="QPSO iterations")
    parser.add_argument("--verbose", action="store_true", help="per-iteration trace")
    args = parser.parse_args()

    center = tuple(args.center) if args.center else None
    info = load_osm_network(center=center, dist=args.dist,
                            physical_capacity=args.physical_capacity)
    G = info["G"]
    print("=== Real road network (OSM) ===")
    print(f"center={info['center']} dist={info['dist_m']}m  crs={info['crs']}")
    print(f"raw: {info['raw_nodes']} nodes / {info['raw_edges']} edges  ->  "
          f"adapted: {info['nodes']} nodes / {info['edges']} directed edges")

    apply_random_congestion(G, variance_level=0.3, seed=42)
    print(f"Congestion variance: {congestion_variance(G):.5f}")

    if args.od == "realistic":
        from real_od_generator import create_realistic_vehicles
        vehicles = create_realistic_vehicles(
            G, center=center if center is not None else (26.9124, 75.7873),
            num_vehicles=args.vehicles, seed=1,
        )
        print(f"Vehicles: {len(vehicles)} (realistic O-D from OSM land-use)")
    else:
        vehicles = create_vehicles(G, num_vehicles=args.vehicles, hub_ratio=1.5, seed=1)
        print(f"Vehicles: {len(vehicles)} (hub_ratio=1.5)")

    if args.physical_capacity:
        caps = [float(d["capacity"]) for _, _, d in G.edges(data=True)]
        print(f"Physical capacities: min {min(caps):.0f} / median {sorted(caps)[len(caps)//2]:.0f} "
              f"/ max {max(caps):.0f} veh/h  (N/3 = {len(vehicles) / 3:.0f})")

    result = build_candidate_edge_set(G, vehicles, verbose=False)
    candidate = result["candidate_edges"]
    feasible = [v for v in vehicles if result["per_vehicle_paths"][v["id"]]]
    dropped = len(vehicles) - len(feasible)
    if dropped:
        print(f"Dropped {dropped} vehicle(s) with no directed O-D path "
              f"(one-way streets); {len(feasible)} remain")
    vehicles = feasible
    cap = len(vehicles) / 3.0

    routes_base, fit_base = evaluate(G, vehicles, {}, candidate)
    print(f"Candidate edges: {len(candidate)} / {G.number_of_edges()} "
          f"(reduction_ratio={result['reduction_ratio']:.3f})")
    print(f"Identity-bias fitness (static): {fit_base:.2f}")

    fit_ue = fitness_with_feedback(G, routes_base, capacity=cap)
    print(f"Greedy flow fitness (User-Equilibrium): {fit_ue:.2f}")

    swarm_flow = run_qpso(
        G, vehicles, candidate,
        num_particles=args.pop, iterations=args.iters, seed=42,
        scoring="flow", capacity=cap, verbose=args.verbose,
    )
    print(f"\nQPSO best fitness (flow):  {swarm_flow['gbest_fitness']:.2f}")
    print(f"Flow baseline beaten:      {swarm_flow['gbest_fitness'] < fit_ue}")
    print(f"Improvement: {(1 - swarm_flow['gbest_fitness'] / fit_ue) * 100:.2f}%")
    print(f"Elapsed: {swarm_flow['elapsed_sec']:.3f}s")

    # sanity: a real shortest path on the adapted map
    sample = route_all_vehicles(G, vehicles, {}, candidate)
    first = sample[1]["path"]
    print(f"\nSample routed path (vehicle 1, {len(first)} nodes): "
          f"{first[:6]}{'...' if len(first) > 6 else ''}")
    print(f"  true_cost = {path_true_cost(G, first):.2f} m")