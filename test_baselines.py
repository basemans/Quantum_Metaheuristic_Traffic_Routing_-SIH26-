"""
test_baselines.py

Assertion-based tests for the classical metaheuristic baselines
(methodology Section 8): run_pso, run_ga. Run with:

    python test_baselines.py

Guarantees verified (identical fairness conventions to run_qpso):
  1. Every method's best fitness is never worse than the greedy UE baseline
     (identity-route seeding).
  2. Best-fitness histories are monotone non-increasing.
  3. Bias-space methods (PSO, GA) keep every bias within [bias_min, bias_max].
  4. All methods terminate quickly and return the run_qpso result shape.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from network_generator import create_grid_network
from vehicle_generator import create_vehicles
from traffic_simulation import apply_random_congestion
from candidate_set import build_candidate_edge_set
from decode import route_all_vehicles, score_function
from baselines import run_ga, run_pso

CAP_RATIO = 3.0


def setup(rows=6, cols=6, num_vehicles=10, seed=7):
    G = create_grid_network(rows=rows, cols=cols, spacing=2000.0)
    apply_random_congestion(G, variance_level=0.3, seed=seed)
    vehicles = create_vehicles(G, num_vehicles=num_vehicles, hub_ratio=1.5, seed=seed)
    result = build_candidate_edge_set(G, vehicles, verbose=False)
    return G, vehicles, result


def ue_fitness(G, vehicles, candidate):
    score = score_function(G, scoring="flow", capacity=len(vehicles) / CAP_RATIO)
    return score(route_all_vehicles(G, vehicles, {}, candidate))


def check_shape_and_history(res, name):
    for key in ("gbest", "gbest_fitness", "gbest_routes", "history",
                "mean_history", "elapsed_sec"):
        assert key in res, f"{name}: missing result key {key}"
    h = res["history"]
    for a, b in zip(h, h[1:]):
        assert b <= a + 1e-9, f"{name}: history increased {a} -> {b}"


def test_pso_flow_mode():
    G, vehicles, result = setup(seed=4)
    candidate = result["candidate_edges"]
    fit_ue = ue_fitness(G, vehicles, candidate)
    res = run_pso(G, vehicles, candidate, num_particles=12, iterations=15,
                  seed=4, scoring="flow", capacity=len(vehicles) / CAP_RATIO)
    check_shape_and_history(res, "pso")
    assert res["gbest_fitness"] <= fit_ue + 1e-9
    for b in res["gbest"].values():
        assert 0.5 <= b <= 1.5, f"pso bias out of range: {b}"
    print(f"ok 1 - PSO (w/c1/c2) never worse than UE, biases bounded, monotone")


def test_ga_flow_mode():
    G, vehicles, result = setup(seed=5)
    candidate = result["candidate_edges"]
    fit_ue = ue_fitness(G, vehicles, candidate)
    res = run_ga(G, vehicles, candidate, pop_size=12, generations=15,
                 seed=5, scoring="flow", capacity=len(vehicles) / CAP_RATIO)
    check_shape_and_history(res, "ga")
    assert res["gbest_fitness"] <= fit_ue + 1e-9
    for b in res["gbest"].values():
        assert 0.5 <= b <= 1.5, f"ga bias out of range: {b}"
    print("ok 2 - GA (BLX-alpha) never worse than UE, biases bounded, monotone")


def test_run_qpso_uses_shared_scorer():
    # ensure qpso_core still exposes the same result contract through the
    # shared score_function (regression after refactor)
    from qpso_core import run_qpso
    G, vehicles, result = setup(seed=8)
    candidate = result["candidate_edges"]
    res = run_qpso(G, vehicles, candidate, num_particles=8, iterations=6,
                   seed=8, scoring="flow", capacity=len(vehicles) / CAP_RATIO)
    check_shape_and_history(res, "qpso")
    print("ok 3 - qpso_core (refactored) still returns standard result shape")


if __name__ == "__main__":
    test_pso_flow_mode()
    test_ga_flow_mode()
    test_run_qpso_uses_shared_scorer()
    print("\nALL BASELINE TESTS PASSED")