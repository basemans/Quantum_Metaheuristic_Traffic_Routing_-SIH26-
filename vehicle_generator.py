"""
vehicle_generator.py

Generates vehicle (start, destination) pairs as a BLEND of two underlying
patterns, controlled by a single ratio parameter:

- hub-based : vehicles concentrated around a small number of hub regions,
              modeling realistic delivery-fleet / district-based demand.
              Best-case for candidate-edge-set reduction.

- random    : uniform random pairs across the whole network.
              Worst-case for candidate-edge-set reduction.

`hub_ratio` sets the mix, e.g. hub_ratio=1.5 (the default) means 1.5 hub-based
vehicles for every 1 random vehicle (60% hub-based / 40% random overall).
hub_ratio=0 -> fully random. A very large hub_ratio -> effectively fully
hub-based. This directly parameterizes the spatial-locality assumption
instead of forcing an all-or-nothing choice between two separate modes.
"""

import random
import math
import networkx as nx


def create_vehicles_random(G: nx.DiGraph, num_vehicles: int, seed: int = None) -> list:
    """Uniform-random (start, destination) pairs across all nodes."""
    rng = random.Random(seed)
    nodes = list(G.nodes)
    vehicles = []

    for i in range(num_vehicles):
        start = rng.choice(nodes)
        destination = rng.choice(nodes)
        while destination == start:
            destination = rng.choice(nodes)

        vehicles.append({
            "id": i + 1,
            "start": start,
            "destination": destination
        })

    return vehicles


def create_vehicles_clustered(
    G: nx.DiGraph,
    num_vehicles: int,
    num_hubs: int = 3,
    hub_radius: float = 1.5,
    seed: int = None
) -> list:
    """
    Vehicles drawn from a small number of hub regions.

    Parameters
    ----------
    num_hubs   : number of district "centers" to place on the network
    hub_radius : radius (in units of grid spacing) around each hub center
                 within which nodes are considered "in" that hub
    """
    rng = random.Random(seed)
    nodes = list(G.nodes)

    if num_hubs < 1:
        raise ValueError("num_hubs must be >= 1")
    if num_hubs > len(nodes):
        raise ValueError("num_hubs cannot exceed number of nodes")

    # Estimate grid spacing from the network so hub_radius is meaningful
    xs = sorted(set(G.nodes[n]["x"] for n in nodes))
    spacing = xs[1] - xs[0] if len(xs) > 1 else 1.0
    radius_dist = hub_radius * spacing

    # Pick hub centers among existing nodes
    hub_centers = rng.sample(nodes, num_hubs)

    # Assign every node to its nearest hub, if within radius; else "unclustered"
    def dist(a, b):
        x1, y1 = G.nodes[a]["x"], G.nodes[a]["y"]
        x2, y2 = G.nodes[b]["x"], G.nodes[b]["y"]
        return math.hypot(x2 - x1, y2 - y1)

    hub_members = {hub: [] for hub in hub_centers}
    for n in nodes:
        nearest_hub = min(hub_centers, key=lambda h: dist(h, n))
        if dist(nearest_hub, n) <= radius_dist:
            hub_members[nearest_hub].append(n)

    # Ensure every hub has at least its own center as a member
    for hub in hub_centers:
        if not hub_members[hub]:
            hub_members[hub] = [hub]

    vehicles = []
    for i in range(num_vehicles):
        # Pick two (possibly same) hubs for start/destination clusters -
        # allows both intra-hub trips and inter-hub trips
        start_hub, dest_hub = rng.sample(hub_centers, 2) if num_hubs > 1 else (hub_centers[0], hub_centers[0])

        start = rng.choice(hub_members[start_hub])
        candidates = [n for n in hub_members[dest_hub] if n != start]
        if candidates:
            destination = rng.choice(candidates)
        else:
            # Dest hub has no node distinct from start (e.g. num_hubs=1
            # with a single-member hub): fall back to any other graph node
            # instead of looping forever.
            others = [n for n in nodes if n != start]
            if not others:
                raise ValueError("Cannot generate distinct start/destination: graph has fewer than 2 nodes")
            destination = rng.choice(others)

        vehicles.append({
            "id": i + 1,
            "start": start,
            "destination": destination
        })

    return vehicles


def create_vehicles(
    G: nx.DiGraph,
    num_vehicles: int,
    hub_ratio: float = 1.5,
    num_hubs: int = 3,
    hub_radius: float = 1.5,
    seed: int = None
) -> list:
    """
    Generates a blended vehicle set: hub_ratio hub-based vehicles for every
    1 random vehicle.

    Parameters
    ----------
    hub_ratio  : ratio of hub-based : random vehicles. Default 1.5 (60%
                 hub-based / 40% random). 0 = fully random. Larger values
                 push toward fully hub-based.
    num_hubs   : number of hub regions to place (only affects the hub-based
                 portion; ignored if hub_ratio == 0)
    hub_radius : radius (in units of grid spacing) defining each hub's extent
    """
    if hub_ratio < 0:
        raise ValueError("hub_ratio must be >= 0")

    rng_seed_split = None if seed is None else seed  # keep deterministic but distinct sub-seeds below

    hub_fraction = hub_ratio / (hub_ratio + 1)
    num_hub_vehicles = round(num_vehicles * hub_fraction)
    num_random_vehicles = num_vehicles - num_hub_vehicles

    vehicles = []

    if num_hub_vehicles > 0:
        hub_vehicles = create_vehicles_clustered(
            G, num_hub_vehicles,
            num_hubs=num_hubs, hub_radius=hub_radius,
            seed=seed
        )
        vehicles.extend(hub_vehicles)

    if num_random_vehicles > 0:
        random_seed = None if seed is None else seed + 1  # distinct stream from hub RNG
        random_vehicles = create_vehicles_random(G, num_random_vehicles, seed=random_seed)
        vehicles.extend(random_vehicles)

    # Reassign sequential ids across the combined set
    for i, v in enumerate(vehicles):
        v["id"] = i + 1

    return vehicles


if __name__ == "__main__":
    from network_generator import create_grid_network

    G = create_grid_network(rows=6, cols=6, spacing=2000.0)

    print("=== Default blend (hub_ratio=1.5, ~60% hub / 40% random) ===")
    vehicles = create_vehicles(G, num_vehicles=10, seed=42)
    for v in vehicles:
        print(v)

    print("\n=== Fully random (hub_ratio=0) ===")
    vehicles_random = create_vehicles(G, num_vehicles=10, hub_ratio=0, seed=42)
    for v in vehicles_random:
        print(v)

    print("\n=== Mostly hub-based (hub_ratio=9, ~90% hub / 10% random) ===")
    vehicles_hub = create_vehicles(G, num_vehicles=10, hub_ratio=9, num_hubs=3, hub_radius=1.2, seed=42)
    for v in vehicles_hub:
        print(v)
