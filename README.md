# Quantum_Metaheuristic_Traffic_Routing_-SIH26-

Quantum-Inspired Particle Swarm Optimization (QPSO) for coordinated traffic routing on synthetic grid networks, benchmarked against classical PSO and a genetic algorithm (GA) under flow-dependent (BPR) congestion. An optional real-world stage ingests small OpenStreetMap districts and runs the same pipeline unchanged on real road geometry.

This README explains **what every Python file in the project does**: its role in the pipeline, its inputs, what it computes, and what it returns. Formal definitions and derivations live in [`docs/MATHEMATICAL_GUIDE.md`](docs/MATHEMATICAL_GUIDE.md); the project overview, benchmark results, and presentation notes live in [`docs/PROJECT_AND_PRESENTATION.md`](docs/PROJECT_AND_PRESENTATION.md); the full experimental log (what was tried, what failed, all measured results) lives in [`docs/RESEARCH_DOCUMENTATION.md`](docs/RESEARCH_DOCUMENTATION.md). The behavior described below is verified by the assertion-based test suites (`test_*.py`).

---

## Pipeline overview

The optimization pipeline is built in five stages plus a benchmarking layer, with an optional real-world front end. The dependency order below is also the order in which the files should be read:

```
network_generator.py ──> vehicle_generator.py ──> traffic_simulation.py
        │                        │                        │
        └─────────── candidate_set.py (Phase 0) ───────────┘
                            │
                     decode.py (decoder + fitness)
                            │
              qpso_core.py (QPSO)  |  baselines.py (PSO, GA)
                            │
                   benchmark.py (harness)

real-world front end (replaces grid + synthetic demand):
      real_network_adapter.py  ──>  real_od_generator.py
```

1. **Network** (`network_generator.py`) — builds the road graph (nodes + directed edges). *Real-world alternative:* `real_network_adapter.py` downloads an OSM district in the same graph contract.
2. **Demand** (`vehicle_generator.py`) — creates the `(start, destination)` pairs. *Real-world alternative:* `real_od_generator.py` derives Jaipur-specific trips from real land use.
3. **Traffic** (`traffic_simulation.py`) — perturbs edges with static congestion, and later prices link flows (BPR).
4. **Phase 0** (`candidate_set.py`) — shrinks the search space to a per-vehicle candidate edge set.
5. **Decode + fitness** (`decode.py`) — turns a bias vector into real routes and scores them.
6. **Optimizers** (`qpso_core.py`, `baselines.py`) — search the bias space.
7. **Benchmark** (`benchmark.py`) — the forward sweep comparing all optimizers.

---

## 1. `network_generator.py` — road network construction

**Role.** Manufactures the synthetic street network: a rectangular grid of intersections connected by directed road segments. It is parameterized, fully deterministic for given inputs, and requires only `networkx` (no `osmnx`/`geopandas`).

**Functions.**

- `create_grid_network(rows, cols, spacing=2000.0) -> networkx.DiGraph`
  - **Inputs:** `rows`, `cols` (grid dimensions, both `>= 1`), `spacing` (meters between adjacent nodes).
  - **Actions:** creates nodes numbered in row-major order `id(r, c) = r·cols + c + 1`, placed at `(x, y) = (c·spacing, r·spacing)`; connects every orthogonal neighbor pair with **two directed edges** carrying attributes `base_length` (Euclidean distance = `spacing` on this grid) and `congestion_factor` (initialized to `1.0`).
  - **Output:** the graph. Edge count `|E| = 4·rows·cols − 2·rows − 2·cols` (e.g. 3×3 → 24, 6×6 → 120).
- `true_cost(G, u, v) -> float` — the cost of traversing an edge: `base_length(u,v) × congestion_factor(u,v)`. This is the **only** cost the fitness function may see; the bias vector (see `decode.py`) is layered on top purely for search.
- `path_true_cost(G, path) -> float` — sum of `true_cost` over a node-sequence path.

**Running it (`__main__`):** prints a 3×3 sanity check — `Nodes: 9`, `Edges: 24`, then the first five edges with their `base_length` (= `2000.0`) and `congestion_factor` (`1.0`).

---

## 2. `vehicle_generator.py` — demand generation

**Role.** Produces the transportation demand: a list of vehicle dictionaries `{"id", "start", "destination"}`. Demand is a **blend** of two spatial patterns to parameterize the locality assumption:

- **hub-based** — trips concentrated around a few district centers (good case for Phase-0 reduction);
- **random** — uniform-random pairs across the whole network (worst case).

**Functions.**

- `create_vehicles_random(G, num_vehicles, seed=None) -> list` — uniform-random `start`/`destination` across all nodes, resampling until `destination != start`.
- `create_vehicles_clustered(G, num_vehicles, num_hubs=3, hub_radius=1.5, seed=None) -> list`
  - **Actions:** estimates grid spacing from the node `x` coordinates; samples `num_hubs` hub centers; assigns each node to its nearest hub if within `hub_radius` (in units of grid spacing, physically `hub_radius × spacing`); generates `start` from one hub and `destination` from another (inter-hub trips) — or the same hub when `num_hubs == 1` (intra-hub). If the destination hub has no node distinct from `start`, it falls back to any other graph node.
- `create_vehicles(G, num_vehicles, hub_ratio=1.5, num_hubs=3, hub_radius=1.5, seed=None) -> list`
  - **The entry point.** `hub_ratio` is the ratio of hub-based to random vehicles; `hub_fraction = hub_ratio/(hub_ratio+1)` so `hub_ratio=1.5` → 60% hub-based. The hub batch uses `seed`, the random batch `seed + 1` (separate RNG streams), then ids are reassigned 1…N.

**Running it (`__main__`):** prints three labeled lists of 10 vehicle dicts — default blend (`hub_ratio=1.5`), fully random (`hub_ratio=0`), and mostly hub-based (`hub_ratio=9`).

---

## 3. `traffic_simulation.py` — static congestion + flow feedback

**Role.** Two congestion layers, both keeping every `congestion_factor >= 1.0` (congestion is a penalty; negative values are impossible).

**Static base (preprocessing).**

- `apply_random_congestion(G, variance_level=0.2, seed=None) -> None` — sets `congestion_factor(u,v) = 1 + |Z(u,v)|` per edge with `Z ~ Normal(0, variance_level)` (a folded normal; `variance_level=0` ⇒ free flow at exactly 1.0). Mutates `G` **in place**.
- `congestion_variance(G) -> float` — the sample variance of all static `congestion_factor` values across edges (`statistics.variance`, `0.0` if fewer than 2 edges). This is the signal Phase 0's dynamic `k` reads.

**Flow feedback (BPR volume–delay, used after routing).**

- `route_flows(routes) -> dict` — counts, for each directed edge, how many routed vehicles traverse it: maps `routes` to `{(u, v): count}`.
- `flow_load_multiplier(flow, capacity, alpha=0.15, beta=4.0) -> float` — the BPR multiplier `m(f) = 1 + α·(f/c)^β`; `m = 1.0` at zero flow, superlinear growth as flow approaches capacity.
- `flow_edge_cost(G, u, v, flow, capacity, alpha=0.15, beta=4.0) -> float` — `base_length × base_congestion_factor × m(flow)`. **Capacity resolution:** if the edge carries its own `capacity` attribute (set by `real_network_adapter.attach_physical_capacity` from OSM lanes × road class), that physical value overrides the scalar `capacity` argument, which remains the fallback for synthetic grids (where `capacity = Nᵥ/3`).

**Running it (`__main__`):** on a 4×4 grid prints `variance = 0.0` before congestion, then `variance = 0.0233…` after `variance_level=0.3`, plus five sample edge `congestion_factor` values.

---

## 4. `candidate_set.py` — Phase 0 search-space reduction

**Role.** Computes the **candidate edge set** — the union, over all vehicles, of the edges appearing on each vehicle's `k` shortest paths. This is the subgraph (and bias-vector dimension) that the optimizers and decoder operate on, instead of the full graph.

**Functions.**

- `dynamic_k(congestion_var, k_min=5, k_max=15, variance_at_kmax=0.1) -> int` — maps congestion variance to how many paths to precompute per vehicle: `k = k_min + min(var/variance_at_kmax, 1)·(k_max − k_min)`, rounded; `k = k_min` if `var <= 0`. More volatile congestion ⇒ more alternates (placeholder mapping — an open question).
- `vehicle_k_shortest_paths(G, start, destination, k) -> list` — up to `k` loopless shortest paths ranked by `true_cost`, via `networkx.shortest_simple_paths` (Yen's algorithm).
- `edges_in_path(path) -> set` — `{(path[i], path[i+1])}`.
- `build_candidate_edge_set(G, vehicles, k_min, k_max, variance_at_kmax, full_map_threshold=0.7, verbose=True) -> dict`
  - **Returns:** `candidate_edges` (set of `(u,v)`), `k_used`, `per_vehicle_paths` (`{id: [paths]}`), `reduction_ratio = |candidate_edges| / |E|`, and `full_map_fallback` (`True` iff the fallback cut in). Vehicles with no path are reported as unreachable.
  - **Guarantee:** every vehicle's own shortest path is always among its `k` paths, so identity-bias routing over the candidate subgraph equals un-restricted full-graph routing — the reduction never makes the greedy baseline worse.
  - **Full-map fallback:** under dense demand the union can cover most of the graph (the reduction buys nothing). If `reduction_ratio ≥ full_map_threshold` (default `0.7`), the candidate set is replaced by **every edge** of the graph, `reduction_ratio` is reported as `1.0`, and routing/search proceed on the full map.

**Running it (`__main__`):** on a 6×6 grid with `variance_level=0.3` prints, per demand mode, `Congestion variance: 0.02684 -> dynamic k = 8` and the candidate-edge counts, then a `COMPARISON` block: random `reduction_ratio = 0.700` (84/120), clustered `0.417` (50/120).

---

## 5. `decode.py` — decoder, solvers, and fitness

**Role.** The **decode layer**: converts a bias vector over `candidate_edges` into a full route set, and evaluates that route set under either scoring objective. The bias is a **search handle**; fitness is always computed on **unbiased** `true_cost`.

**Reweighting.**

- `effective_cost(G, u, v, bias=None) -> float` — `true_cost(u,v) × bias.get((u,v), 1.0)`. Missing edges default to bias `1.0`.

**Solvers** (both restricted to `allowed_edges` and implemented on `heapq`; `allowed_edges=None` ⇒ full graph).

- `dijkstra(G, start, target, bias=None, allowed_edges=None) -> (path, effective_cost)` — reference shortest-path.
- `a_star(G, start, target, bias=None, allowed_edges=None) -> (path, effective_cost)` — A* with heuristic `h(n) = euclidean(n, target) × κ`, where `κ` is the cheapest effective cost per meter anywhere in the allowed subgraph. This heuristic is **admissible and consistent**, so A* provably returns exactly what Dijkstra returns — the two exist as a runtime comparison, not a quality one.

**Routing + scoring.**

- `route_all_vehicles(G, vehicles, bias=None, allowed_edges=None, algorithm="dijkstra") -> dict` — routes every vehicle; returns `{id: {"path", "effective_cost", "true_cost"}}`.
- `fitness(G, routes) -> float` — the **static** objective: sum of unbiased `true_cost` over all vehicles (decomposable; identity routing is optimal).
- `fitness_with_feedback(G, routes, capacity, alpha=0.15, beta=4.0) -> float` — the **flow** objective: system-total travel time where every edge is priced at `flow_edge_cost` given its vehicle flow. Non-decomposable — this is the coordination problem QPSO solves.
- `evaluate(G, vehicles, bias, allowed_edges, algorithm) -> (routes, fitness)` — convenience wrapper.
- `score_function(G, scoring="static", capacity=None, alpha=0.15, beta=4.0)` — returns `routes -> fitness` callable shared by **all** optimizers (apples-to-apples benchmark). Raises if `scoring="flow"` is used without `capacity`.

**Running it (`__main__`):** prints `Candidate edges: 62 (reduction_ratio=0.517)`, the identity-bias fitness under Dijkstra (`80852.60`) and A* (`80852.60`), and `(A* == Dijkstra is expected: True)`.

---

## 6. `qpso_core.py` — Quantum-inspired Particle Swarm Optimization

**Role.** The core optimizer. Searches the bias space `ℝ^D`, `D = |candidate_edges|`, minimizing the chosen fitness. Single-threaded.

**Search design.** A particle's position is a bias vector `{edge: bias}`. Per dimension `j`, per iteration (fresh random draws):

```
φ ~ U(0,1);  p = φ·pbest_i + (1−φ)·gbest          (local attractor)
u ~ U(0,1);  sign ± 1
x_new = p + sign · β(t) · |x − mbest| · ln(1/u)
x_new ← clip(x_new, 0.5, 1.5)
```

- `mbest` = elementwise mean of all personal bests;
- `β(t)` = contraction–expansion coefficient, annealed linearly `beta_start=1.0 → beta_end=0.5` (wide early exploration, focused late exploitation);
- **no velocity term** — this fully replaces PSO's `w, c1, c2` update.

**Design guarantees (enforced by validation).** `bias_min ≤ 1.0 ≤ bias_max` is required so 1.0 is reachable; particle 0 is **seeded at identity bias** (all 1.0), so its first decayed fitness equals plain shortest-path routing and `gbest` is provably never worse than the greedy baseline; `history` is monotone non-increasing.

**Interface.** `run_qpso(G, vehicles, candidate_edges, num_particles=20, iterations=30, beta_start=1.0, beta_end=0.5, bias_min=0.5, bias_max=1.5, seed=None, algorithm="dijkstra", scoring="static", capacity=8.0, alpha=0.15, beta=4.0, verbose=False) -> dict`.

**Returns:** `gbest` (edge→bias), `gbest_fitness`, `gbest_routes` (decoded), `history` (best per iteration), `mean_history` (mean swarm fitness per iteration), `elapsed_sec` (excluding init).

**Running it (`__main__`):** prints the static identity baseline (`80852.60`) and QPSO static tie, then routes into the flow regime: greedy User-Equilibrium flow fitness `84478.60`, a 40-iteration per-iteration trace (`iter  n/40: best=… mean=… beta=…`), and the finale `QPSO best fitness (flow): 81564.31`, `Improvement: 3.45%`, `Elapsed: ~0.33s`.

---

## 7. `baselines.py` — classical PSO and GA

**Role.** Comparative baselines over the *same* bias space, *same* scorer (`decode.score_function`), and the *same* fairness rules (identity-seeded individual 0, equal population/iteration budget, deterministic under `seed`, identical result shape) so the benchmark is apples-to-apples.

- `run_pso(G, vehicles, candidate_edges, num_particles=20, iterations=30, c1=2.0, c2=2.0, inertia_start=0.9, inertia_end=0.4, …) -> dict` — classical velocity update `v ← w(t)·v + c1·r1·(pbest−x) + c2·r2·(gbest−x)` with linear inertia schedule `0.9 → 0.4`, positions clipped to `[0.5, 1.5]`.
- `run_ga(G, vehicles, candidate_edges, pop_size=20, generations=30, crossover_prob=0.8, mutation_prob=0.1, blx_alpha=0.5, tournament_size=3, elite=1, …) -> dict` — real-coded GA: elitism, binary tournament selection, BLX-α crossover, Gaussian mutation with `σ = (bias_max − bias_min)/10`, all clipped to `[0.5, 1.5]`.

~~An Ant Colony System (ACO) baseline was implemented, benchmarked, and then **removed**: its route-construction search recovered ~0% on every default-sweep config (best isolated run 0.68% at 2.5× budget, seed-dependent) because coordinated flow-diversions are vanishingly rare points in its construction space — see `docs/PROJECT_AND_PRESENTATION.md` §4.10.~~

**Running it:** has **no** `__main__` block — it is imported by `benchmark.py` and exercised by `test_baselines.py`; running the file directly produces no output.

---

## 8. `benchmark.py` — benchmarking harness

**Role.** Runs the full pipeline (files 1–5) across a configured sweep of configurations and compares QPSO, PSO, and GA under flow-feedback (BPR) scoring at **equal budget** (`pop=16`, `iters=20` default; `capacity = vehicles/3`).

**Sweep dimensions** (Cartesian product, default = 16 configs): network size `{(4,4), (6,6)}`, vehicle count `{10, 20}`, demand mode `hub_ratio ∈ {0 (random), 9 (clustered)}`, congestion variance `{0.1, 0.5}`.

**Key functions.**

- `run_single(cfg, methods, pop, iters, seed_base)` — one config × all methods over a **shared** network, vehicles, candidate set, and scorer; returns per-method rows (final fitness, `improv_pct = (ue−final)/ue×100`, runtime) and convergence curves.
- `exact_ue_fitness(...)` — the greedy User-Equilibrium baseline (identity-bias routing scored under the same objective).
- `aggregate_best_method(rows, conv)` — adds `gap_to_best_pct` and counts **strict** per-config wins by improvement.
- `main()` — CLI with `--quick`, `--out`, `--pop`, `--iters`, `--methods`, `--size`, `--vehicles`, `--hub`, `--var`; writes `summary.csv`, `convergence.csv`, `scaling.csv`, `reduction.csv` to `bench_out/` and prints the summary table + win/loss.

**Running it:** `python benchmark.py` → `Sweep: 16 configs x 3 methods (pop=16, iters=20)`, the per-config improvement table, strict-winner counts (`qpso 3 / pso 3 / ga 1`, 9 exact ties), and `Outputs written to …\bench_out`. `python benchmark.py --quick` runs a 2-config smoke sweep (pop=8, iters=10).

---

## 9. `real_network_adapter.py` — real-road ingestion (OSM)

**Role.** Replaces the synthetic grid with a **real street network** downloaded from OpenStreetMap, converted into the exact same graph contract as `create_grid_network` (`nodes` with `x`/`y` in projected metres; directed edges with `base_length` and `congestion_factor`), so the whole pipeline (4–8 above) runs unchanged. Requires `osmnx` + `geopandas`/`shapely`/`pyproj` (extra dependencies over the synthetic path).

**Conversion choices** (documented in the file docstring):
- `network_type='drive'` — drivable roads only; `simplify=True` contracts OSM geometry into intersection nodes;
- parallel/duplicate OSM edges merged to the shortest `length`; **zero-length edges dropped** (otherwise `true_cost / base_length` in the A*-heuristic scale would divide by zero);
- largest **weakly-connected** component retained (a skeleton path exists between any two nodes in the undirected sense);
- road metadata kept (`highway`, `lanes`, `oneway`) for the physical-capacity pass.

**Functions.**

- `_to_simple_digraph(Gp) -> nx.DiGraph` — the OSM → project DiGraph conversion.
- `attach_physical_capacity(G) -> None` — adds a per-edge `capacity` (vehicles/hour) = `lanes × per-lane(road class)`, **halved for two-way streets**. `flow_edge_cost` (see §3) honours this attribute over the scalar `Nᵥ/3` fallback — this is what makes congestion meaningful at realistic per-hour demand instead of scaling away.
- `load_osm_network(center=None, dist=1200.0, network_type='drive', simplify=True, physical_capacity=False) -> dict` — the entry point; returns `G`, `center`, `dist_m`, raw/adapted node+edge counts, and the CRS. Default center is Jaipur city centre `(26.9124, 75.7873)`.

**Running it (`__main__`):** a full end-to-end pipeline demo on real roads. Flags: `--center LAT LON`, `--dist`, `--vehicles`, `--od {synthetic,realistic}`, `--physical-capacity`, `--pop`, `--iters`, `--verbose`. Prints the network stats, demand summary, Phase-0 reduction, greedy UE flow fitness, and the QPSO improvement. Example (the physically consistent realistic test, see the research doc):

```
python real_network_adapter.py --vehicles 5000 --od realistic --physical-capacity --pop 8 --iters 10
```

---

## 10. `real_od_generator.py` — realistic Jaipur-specific O–D demand

**Role.** Produces demand that means something physically: trips from **residential** areas to **education/office/commercial** destinations, instead of uniform random pairs. Jaipur-specific by design, hence kept in its own module rather than `vehicle_generator.py`.

**Functions.**

- `_category_nodes(G, center, dist, crs, lonlat, tags, label) -> list` — fetches OSM features matching a tag query (e.g. `landuse: residential`, or `amenity: school/college/university/…`), snaps each feature (`representative_point()` for polygons) to its **nearest road node** via haversine over the inverse-projected node set; falls back to all nodes if the district has none of that category.
- `create_realistic_vehicles(G, center, dist=1000.0, num_vehicles=1000, crs='EPSG:32643', seed=None) -> list` — returns the standard contract `{"id", "start", "destination"}` with origins sampled from residential nodes and destinations from education/office/commercial nodes (distinct from origin). Measured on the mapped Jaipur district: 11 residential gateway nodes, 63 destination nodes.

Note: with real one-way streets, ~3–12% of sampled pairs are directed-unreachable; the adapter demo reports and drops those (the flow objective requires each demand to be achievable).

---

## 11. Test files — no-framework assertion suites

All are run directly (`python test_*.py`) and print `ok N - …` per check followed by a PASSED banner. They construct a fresh network/vehicles/candidate set per test with fixed seeds, so failures are deterministic.

- **`test_decode.py`** (5 checks) — `effective_cost` multiplies bias on top of `true_cost`; identity bias on the candidate subgraph == full-graph shortest paths; A* == Dijkstra under identity and random bias; fitness == sum of unbiased `true_cost`; `reduction_ratio ∈ (0, 1]`.
- **`test_qpso.py`** (9 checks) — `gbest` never worse than the baseline; `history` monotone; biases within `[0.5, 1.5]`; convergence; mean swarm fitness ≥ best; `route_flows` counts shared edges; flow fitness reduces to static as `capacity → ∞`; flow feedback is non-decomposable (greedy flow cost > static); QPSO beats greedy UE flow on the seed-0 instance.
- **`test_baselines.py`** (3 checks) — PSO in flow mode meets the guarantee (never worse than UE, biases bounded, monotone history); GA same; `run_qpso` returns the standard result shape after the shared-scorer refactor.

---

## How to run

- **Dependencies (synthetic path):** Python 3.13+ and `networkx` (`pip install networkx`).
- **Dependencies (real-road path):** `osmnx`, `geopandas`, `shapely`, `pyproj` (`pip install osmnx` pulls these transitively). Roads are fetched live from OpenStreetMap on first use.
- **Pipeline demo (synthetic):** `python qpso_core.py` (it builds Everything: network → vehicles → congestion → Phase 0 → decode → QPSO).
- **Pipeline demo (real roads):** `python real_network_adapter.py --od realistic --physical-capacity --vehicles 5000 --pop 8 --iters 10` (or `--od synthetic --vehicles 10` for the small showcase).
- **Verify:** `python test_decode.py`, `python test_qpso.py`, `python test_baselines.py`.
- **Benchmark:** `python benchmark.py` (full sweep) or `python benchmark.py --quick` (smoke).
- **Docs:** see [`docs/MATHEMATICAL_GUIDE.md`](docs/MATHEMATICAL_GUIDE.md) for derivations and the worked example, `docs/PROJECT_AND_PRESENTATION.md` for results and presentation notes, and `docs/RESEARCH_DOCUMENTATION.md` for the full experimental history.