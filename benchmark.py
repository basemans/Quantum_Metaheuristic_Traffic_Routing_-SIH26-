"""
benchmark.py

Benchmarking harness (methodology Section 8). Runs the full pipeline across
sweep dimensions and compares QPSO against the classical metaheuristics
(classical PSO, GA) and the exact greedy User-Equilibrium baseline
(Dijkstra identity routing).

Sweep dimensions (all parameterized):
    network size (rows x cols), vehicle count, demand mode
    (hub_ratio: 0 = random, 9 = clustered), congestion variance level

Objective: flow-feedback (BPR) system-total travel time by default - the
non-decomposable setting where coordinated routing can beat greedy.

(An ACO baseline was removed: route-construction search cannot exploit the
coordinated-diversion space -- see baselines.py docstring and docs 4.10.)

Outputs (written to `--out`, default `bench_out/`):
    summary.csv      per config x method: final fitness, improvement vs UE
                     baseline (%), gap to best method (%), runtime, ratio
    convergence.csv  per config x method: (iteration, best fitness) curve
    scaling.csv      mean runtime per method per network size
    reduction.csv    reduction_ratio by demand mode (clustering test)

Usage:
    python benchmark.py           full default sweep
    python benchmark.py --quick   tiny sweep for smoke testing
    python benchmark.py --size 6 6 --vehicles 10 20 --hub 0 9 --var 0.1 0.5
"""

import argparse
import csv
import os
import time

from network_generator import create_grid_network
from vehicle_generator import create_vehicles
from traffic_simulation import apply_random_congestion
from candidate_set import build_candidate_edge_set
from decode import route_all_vehicles, score_function
from qpso_core import run_qpso
from baselines import run_ga, run_pso

METHODS = {
    "qpso": run_qpso,
    "pso": run_pso,
    "ga": run_ga,
}

CAPACITY_RATIO = 3.0  # capacity = num_vehicles / 3 makes BPR flow effects bite


def exact_ue_fitness(G, vehicles, candidate_edges, score):
    """Exact baseline: identity-bias Dijkstra routing scored under the same
    objective as the metaheuristics (greedy User-Equilibrium)."""
    routes = route_all_vehicles(G, vehicles, {}, candidate_edges)
    return score(routes)


def build_configs(sizes, vehicles, hub_ratios, variances):
    return [
        {"rows": r, "cols": c, "vehicles": n, "hub_ratio": h, "variance": v}
        for r, c in sizes
        for n in vehicles
        for h in hub_ratios
        for v in variances
    ]


def run_single(cfg, methods, pop, iters, seed_base):
    """Runs one sweep config across all methods. Shared network, vehicles,
    candidate set, and scorer so every method sees identical inputs."""
    rows, cols = cfg["rows"], cfg["cols"]
    nv, hub, var = cfg["vehicles"], cfg["hub_ratio"], cfg["variance"]

    G = create_grid_network(rows=rows, cols=cols, spacing=2000.0)
    apply_random_congestion(G, variance_level=var, seed=seed_base)
    vehicles = create_vehicles(G, num_vehicles=nv, hub_ratio=hub, seed=seed_base)
    result = build_candidate_edge_set(G, vehicles, verbose=False)
    candidate = result["candidate_edges"]
    capacity = nv / CAPACITY_RATIO

    score = score_function(G, scoring="flow", capacity=capacity)
    fit_ue = exact_ue_fitness(G, vehicles, candidate, score)

    rows_out, conv_out = [], []
    pop_kwargs = {
        "qpso": {"num_particles": pop, "iterations": iters},
        "pso": {"num_particles": pop, "iterations": iters},
        "ga": {"pop_size": pop, "generations": iters},
    }
    for name in methods:
        fn = METHODS[name]
        t0 = time.perf_counter()
        res = fn(
            G, vehicles, candidate,
            seed=seed_base + 1, scoring="flow", capacity=capacity,
            **pop_kwargs[name],
        )
        elapsed = time.perf_counter() - t0
        final = res["gbest_fitness"]
        improv = (fit_ue - final) / fit_ue * 100.0
        rows_out.append({
            "rows": rows, "cols": cols, "vehicles": nv,
            "hub_ratio": hub, "variance": var,
            "reduction_ratio": result["reduction_ratio"],
            "method": name, "final_fitness": round(final, 4),
            "ue_fitness": round(fit_ue, 4),
            "improve_pct": round(improv, 4),
            "runtime_sec": round(elapsed, 4),
        })
        for i, bf in enumerate(res["history"]):
            conv_out.append({
                "rows": rows, "cols": cols, "vehicles": nv,
                "hub_ratio": hub, "variance": var,
                "method": name, "iteration": i, "best_fitness": round(bf, 4),
            })
    return rows_out, conv_out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    header = ("size      veh  hub  var   method  reduce   ue_fit     final    "
              "improv%   runtime")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['rows']}x{r['cols']}     {r['vehicles']:>3}  {r['hub_ratio']:>3}  "
            f"{r['variance']:.1f}  {r['method']:<6}"
            f"  {r['reduction_ratio']:.3f}  {r['ue_fitness']:>9.2f}  "
            f"{r['final_fitness']:>9.2f}  {r['improve_pct']:>7.2f}  "
            f"{r['runtime_sec']:>7.3f}s"
        )


def aggregate_scaling(rows):
    """Mean runtime per method per network size (runtime scaling)."""
    out = []
    sizes = sorted({(r["rows"], r["cols"]) for r in rows})
    for s in sizes:
        for m in METHODS:
            sub = [r for r in rows if r["rows"] == s[0] and r["cols"] == s[1]
                   and r["method"] == m]
            if sub:
                out.append({
                    "size": f"{s[0]}x{s[1]}",
                    "method": m,
                    "mean_runtime_sec": round(sum(r["runtime_sec"] for r in sub) / len(sub), 4),
                    "n_configs": len(sub),
                })
    return out


def aggregate_reduction(rows):
    """Mean reduction_ratio by demand mode (spatial-locality test)."""
    out = []
    for hub in sorted({r["hub_ratio"] for r in rows}):
        sub = [r for r in rows if r["hub_ratio"] == hub]
        out.append({
            "hub_ratio": hub,
            "mode": "clustered" if hub > 0 else "random",
            "mean_reduction_ratio": round(
                sum(r["reduction_ratio"] for r in sub) / len(sub), 4
            ),
            "n_configs": len(sub),
        })
    return out


def aggregate_best_method(rows, conv):
    """Adds gap-to-best column, then counts per-config wins by improvement."""
    win_counts = dict.fromkeys(METHODS, 0)
    per_config = {}
    for r in rows:
        key = (r["rows"], r["cols"], r["vehicles"], r["hub_ratio"], r["variance"])
        per_config.setdefault(key, []).append(r)

    for key, subs in per_config.items():
        best = min(s["final_fitness"] for s in subs)
        top_improv = max(s["improve_pct"] for s in subs)
        top_methods = [s for s in subs if s["improve_pct"] == top_improv]
        for s in subs:
            s["gap_to_best_pct"] = round((s["final_fitness"] - best) / best * 100.0, 4)
        if len(top_methods) == 1:
            win_counts[top_methods[0]["method"]] += 1
    return rows, win_counts


def main():
    parser = argparse.ArgumentParser(description="QPSO benchmarking harness")
    parser.add_argument("--quick", action="store_true", help="tiny smoke sweep")
    parser.add_argument("--out", default=None, help="output directory "
                        "(default: <script_dir>/bench_out)")
    parser.add_argument("--pop", type=int, default=None, help="population/particles/ants")
    parser.add_argument("--iters", type=int, default=None, help="iterations/generations")
    parser.add_argument("--methods", nargs="+", default=list(METHODS),
                        choices=list(METHODS), help="methods to run")
    parser.add_argument("--size", nargs=2, type=int, action="append", metavar=("R", "C"))
    parser.add_argument("--vehicles", nargs="+", type=int)
    parser.add_argument("--hub", nargs="+", type=float)
    parser.add_argument("--var", nargs="+", type=float)
    args = parser.parse_args()
    if args.out is None:
        args.out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_out")

    if args.quick:
        sizes = [(4, 4)]
        vehicles = [10]
        hubs = [0.0, 9.0]
        variances = [0.3]
        pop, iters = 8, 10
    else:
        sizes = args.size if args.size else [(4, 4), (6, 6)]
        vehicles = args.vehicles if args.vehicles else [10, 20]
        hubs = args.hub if args.hub else [0.0, 9.0]
        variances = args.var if args.var else [0.1, 0.5]
        pop = args.pop if args.pop is not None else 16
        iters = args.iters if args.iters is not None else 20

    cfgs = build_configs(sizes, vehicles, hubs, variances)
    print(f"Sweep: {len(cfgs)} configs x {len(args.methods)} methods "
          f"(pop={pop}, iters={iters})\n")

    all_rows, all_conv = [], []
    for ci, cfg in enumerate(cfgs):
        rows, conv = run_single(cfg, args.methods, pop, iters, seed_base=ci * 100)
        all_rows.extend(rows)
        all_conv.extend(conv)

    all_rows, win_counts = aggregate_best_method(all_rows, all_conv)

    os.makedirs(args.out, exist_ok=True)
    write_csv(os.path.join(args.out, "summary.csv"), all_rows)
    write_csv(os.path.join(args.out, "convergence.csv"), all_conv)
    write_csv(os.path.join(args.out, "scaling.csv"), aggregate_scaling(all_rows))
    write_csv(os.path.join(args.out, "reduction.csv"), aggregate_reduction(all_rows))

    print_summary(all_rows)
    print("\nWin/loss (best improvement per config):")
    for m, c in sorted(win_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {m:<6} {c} config(s)")
    print("\nOutputs written to", os.path.abspath(args.out))


if __name__ == "__main__":
    main()