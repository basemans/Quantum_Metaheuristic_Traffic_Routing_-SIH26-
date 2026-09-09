"""
test_decode.py

Assertion-based smoke tests for the decode layer + fitness (methodology
Sections 5-6). No external test framework - run with:

    python test_decode.py

Checks:
  1. effective_cost applies the bias multiplier on top of true_cost.
  2. Identity bias (empty dict) over the candidate subgraph yields the exact
     same routes and costs as full-graph routing.
  3. A* == Dijkstra for both the identity bias and a random non-trivial bias
     (validates the heuristic stays admissible/consistent under reweighting).
  4. Fitness equals the sum of unbiased true_cost.
  5. Phase 0 reduction_ratio sanity: 0 < |candidate_edges| <= |E|.
"""

import math
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from network_generator import create_grid_network, true_cost
from vehicle_generator import create_vehicles
from traffic_simulation import apply_random_congestion
from candidate_set import build_candidate_edge_set
from decode import (
    a_star,
    dijkstra,
    effective_cost,
    evaluate,
    fitness,
    route_all_vehicles,
)


def setup(rows=6, cols=6, num_vehicles=10, seed=7):
    G = create_grid_network(rows=rows, cols=cols, spacing=2000.0)
    apply_random_congestion(G, variance_level=0.3, seed=seed)
    vehicles = create_vehicles(G, num_vehicles=num_vehicles, hub_ratio=1.5, seed=seed)
    result = build_candidate_edge_set(G, vehicles, verbose=False)
    return G, vehicles, result


def test_effective_cost_bias():
    G, _, _ = setup()
    u, v = next(iter(G.edges()))
    base = true_cost(G, u, v)
    assert effective_cost(G, u, v, {}) == base
    assert effective_cost(G, u, v, {(u, v): 2.0}) == 2.0 * base
    assert effective_cost(G, u, v, {}) + effective_cost(G, v, u, {}) > 0
    print("ok 1 - effective_cost applies bias on top of true_cost")


def test_identity_bias_matches_full_graph():
    G, vehicles, result = setup()
    candidate = result["candidate_edges"]
    all_edges = set(G.edges())

    routes_sub = route_all_vehicles(G, vehicles, bias={}, allowed_edges=candidate)
    routes_full = route_all_vehicles(G, vehicles, bias={}, allowed_edges=all_edges)

    assert set(routes_sub) == set(routes_full) == {v["id"] for v in vehicles}
    for vid in routes_sub:
        assert routes_sub[vid]["effective_cost"] == routes_full[vid]["effective_cost"]
        assert routes_sub[vid]["true_cost"] == routes_full[vid]["true_cost"]
        assert routes_sub[vid]["path"] == routes_full[vid]["path"]
    print("ok 2 - identity bias on candidate subgraph == full-graph shortest paths")


def test_astar_equals_dijkstra():
    G, vehicles, result = setup(seed=11)
    candidate = result["candidate_edges"]

    # identity bias
    rd, _ = evaluate(G, vehicles, {}, candidate, algorithm="dijkstra")
    ra, _ = evaluate(G, vehicles, {}, candidate, algorithm="astar")
    assert [rd[k]["path"] for k in rd] == [ra[k]["path"] for k in ra]
    assert fitness(G, rd) == fitness(G, ra)

    # random reweighting (must not break admissibility/consistency)
    rng = random.Random(0)
    bias = {edge: rng.uniform(0.5, 2.0) for edge in candidate}
    rd2, _ = evaluate(G, vehicles, bias, candidate, algorithm="dijkstra")
    ra2, _ = evaluate(G, vehicles, bias, candidate, algorithm="astar")
    for k in rd2:
        assert rd2[k]["effective_cost"] == ra2[k]["effective_cost"], (
            f"vehicle {k}: Dijkstra {rd2[k]['effective_cost']} != A* {ra2[k]['effective_cost']}"
        )
    assert fitness(G, rd2) == fitness(G, ra2)
    print("ok 3 - A* == Dijkstra under identity and random bias")


def test_fitness_is_unbiased_sum():
    G, vehicles, result = setup(seed=3)
    candidate = result["candidate_edges"]
    routes, fit = evaluate(G, vehicles, {}, candidate, algorithm="dijkstra")
    manual = 0.0
    for v in vehicles:
        r = routes[v["id"]]
        path_sum = sum(
            true_cost(G, r["path"][i], r["path"][i + 1])
            for i in range(len(r["path"]) - 1)
        )
        assert path_sum == r["true_cost"]
        manual += path_sum
    assert math.isclose(fit, manual, rel_tol=1e-9)
    assert fit > 0
    print("ok 4 - fitness == sum of unbiased true_cost over all vehicles")


def test_reduction_ratio_sanity():
    G, vehicles, result = setup(seed=5)
    n_edges = G.number_of_edges()
    assert len(result["candidate_edges"]) > 0
    assert len(result["candidate_edges"]) <= n_edges
    assert result["reduction_ratio"] > 0.0
    assert result["reduction_ratio"] <= 1.0
    print(f"ok 5 - reduction_ratio={result['reduction_ratio']:.3f} within (0, 1]")


if __name__ == "__main__":
    test_effective_cost_bias()
    test_identity_bias_matches_full_graph()
    test_astar_equals_dijkstra()
    test_fitness_is_unbiased_sum()
    test_reduction_ratio_sanity()
    print("\nALL TESTS PASSED")