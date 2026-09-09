"""
network_generator.py

Builds a synthetic grid-based transportation network using plain networkx
(no osmnx/geopandas dependency needed - we control every coordinate and
edge ourselves, so those libraries were never doing real work for us).

Each node has (x, y) coordinates.
Each directed edge has:
    - base_length      : Euclidean distance (static, "ground truth" road length)
    - congestion_factor: multiplier >= 1.0, starts at 1.0, perturbed by the
                          traffic simulation layer (see traffic_simulation.py)

true_cost(u, v) = base_length(u, v) * congestion_factor(u, v)
This is the ONLY cost QPSO's fitness function is allowed to see.
The bias vector (added later, in qpso_core.py) is a separate multiplier
layered on top of true_cost purely for search purposes during decode.
"""

import networkx as nx
import math


def create_grid_network(rows: int, cols: int, spacing: float = 2000.0) -> nx.DiGraph:
    """
    Build an R x C grid network.

    Parameters
    ----------
    rows, cols : grid dimensions (e.g. 3x3 = 9 nodes, matches the original
                 synthetic network; try larger values like 8x8, 15x15 for
                 scalability testing)
    spacing    : distance in meters between adjacent grid nodes

    Returns
    -------
    G : networkx.DiGraph
        node attrs: x, y
        edge attrs: base_length, congestion_factor
    """
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be >= 1")

    G = nx.DiGraph()

    # --- Nodes ---
    node_id_of = {}  # (row, col) -> node_id
    node_id = 1
    for r in range(rows):
        for c in range(cols):
            x = c * spacing
            y = r * spacing
            G.add_node(node_id, x=x, y=y)
            node_id_of[(r, c)] = node_id
            node_id += 1

    def euclidean(n1, n2):
        x1, y1 = G.nodes[n1]["x"], G.nodes[n1]["y"]
        x2, y2 = G.nodes[n2]["x"], G.nodes[n2]["y"]
        return math.hypot(x2 - x1, y2 - y1)

    # --- Edges: horizontal and vertical grid connections, both directions ---
    def add_bidirectional_edge(a, b):
        length = euclidean(a, b)
        G.add_edge(a, b, base_length=length, congestion_factor=1.0)
        G.add_edge(b, a, base_length=length, congestion_factor=1.0)

    for r in range(rows):
        for c in range(cols):
            current = node_id_of[(r, c)]

            # horizontal neighbor (right)
            if c + 1 < cols:
                add_bidirectional_edge(current, node_id_of[(r, c + 1)])

            # vertical neighbor (down)
            if r + 1 < rows:
                add_bidirectional_edge(current, node_id_of[(r + 1, c)])

    return G


def true_cost(G: nx.DiGraph, u, v) -> float:
    """The real-world cost of traversing edge (u, v). Used for fitness
    evaluation - must NEVER include the QPSO bias multiplier."""
    data = G.edges[u, v]
    return data["base_length"] * data["congestion_factor"]


def path_true_cost(G: nx.DiGraph, path: list) -> float:
    """Sum of true_cost over a full node-sequence path."""
    return sum(true_cost(G, path[i], path[i + 1]) for i in range(len(path) - 1))


if __name__ == "__main__":
    # Sanity check: rebuild the original 3x3 grid and inspect it
    G = create_grid_network(rows=3, cols=3, spacing=2000.0)
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())
    for u, v, data in list(G.edges(data=True))[:5]:
        print(f"  {u} -> {v}: base_length={data['base_length']:.1f}, "
              f"congestion_factor={data['congestion_factor']}")
