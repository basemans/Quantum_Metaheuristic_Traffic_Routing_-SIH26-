"""
real_od_generator.py

Jaipur-specific realistic origin-destination generation.

Synthetic O-D pairs (vehicle_generator.py) sample uniformly over nodes, which
produces trips with no physical meaning. This module instead grounds demand in
actual OpenStreetMap land use for the mapped Jaipur district:

  ORIGINS/start nodes  : residential land use / residential building
                         footprints (where people live).
  DESTINATION nodes    : education (schools, colleges, universities),
                         office buildings, and commercial/hospital/bank
                         buildings (where people commute to).

Every OSM feature intersection (polygon/point) is snapped to its NEAREST node
of the adapted road network, so each demand endpoint is a real junction on the
routable graph. Vehicle pairs are sampled uniformly among (residential ->
destination) feature nodes, deliberately avoiding a feature node as its own
target.

This grounds the demand in the region's physical geography (morning-peak-ish
commutes), which is exactly the test the user wanted: realistic trips on a
realistic road map, rather than random pairs on an arbitrary grid.

Design notes:
    - Kept fully separate from vehicle_generator.py because it is specific to
      this Jaipur trace (and any OSM area), not part of the synthetic module.
    - Feature queries go through osmnx (cached), returning in WGS84; our graph
      nodes are in projected metres, so snapping is done by inverse-projecting
      the graph nodes to lon/lat once, then haversine over the (small) node
      set - brute force is fine here (hundreds of nodes).
    - If OSM is sparse for the district (no residential/destination features),
      the category falls back to all graph nodes so the run never dies;
      feature counts are always reported.
"""

import math
import random

import networkx as nx
import osmnx as ox
from pyproj import Transformer


# --- OSM tag queries for the two demand categories -------------------------

# Places people live from
_ORIGIN_TAGS = {
    "landuse": ["residential"],
    "building": ["residential", "apartments", "house", "terrace",
                 "detached", "semidetached_house"],
}

# Places people go to (work / study / shop)
_DESTINATION_TAGS = {
    "amenity": ["school", "college", "university", "kindergarten",
                "hospital", "clinic", "bank", "marketplace", "restaurant",
                "cafe", "pharmacy"],
    "office": True,
    "shop": True,
    "building": ["commercial", "office", "school", "college", "university",
                 "hospital"],
    "landuse": ["commercial", "industrial"],
}


def _lonlat_index(G: nx.DiGraph, crs: str) -> dict:
    """Inverse-projects graph nodes (projected metres) to (lon, lat) so OSM
    feature coords (WGS84) can be snapped to them. Returns {node: (lon, lat)}."""
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lonlat = {}
    for n, data in G.nodes(data=True):
        lon, lat = transformer.transform(data["x"], data["y"])
        lonlat[n] = (lon, lat)
    return lonlat


def _nearest_node(lon, lat, lonlat: dict) -> tuple:
    """Haversine-nearest graph node to a WGS84 coordinate."""
    best_node, best_d = None, float("inf")
    lat_r = math.radians(lat)
    for n, (nlon, nlat) in lonlat.items():
        dlat = math.radians(lat - nlat)
        dlon = math.radians(lon - nlon)
        a = math.sin(dlat / 2) ** 2 + math.cos(lat_r) * \
            math.cos(math.radians(nlat)) * math.sin(dlon / 2) ** 2
        d = 2 * 6371000.0 * math.asin(math.sqrt(a))
        if d < best_d:
            best_node, best_d = n, d
    return best_node


def _category_nodes(
    G: nx.DiGraph, center: tuple, dist: float, crs: str,
    lonlat: dict, tags: dict, label: str
) -> list:
    """Fetches OSM features matching `tags` within `dist` of `center`, snaps
    each to its nearest road node, and returns the distinct node list.
    Falls back to all nodes if OSM has no features of that category."""
    features = ox.features_from_point(center_point=center, tags=tags, dist=dist)
    nodes = set()
    for geometry in features.geometry:
        if geometry is None:
            continue
        point = (geometry.representative_point()
                 if geometry.geom_type in ("Polygon", "MultiPolygon")
                 else geometry)
        nodes.add(_nearest_node(point.x, point.y, lonlat))

    print(f"  {label}: {len(features)} OSM feature(s) -> "
          f"{len(nodes)} distinct road node(s)"
          + ("" if nodes else "  [fallback: all nodes]"))
    return list(nodes) if nodes else list(G.nodes)


def create_realistic_vehicles(
    G: nx.DiGraph,
    center: tuple,
    dist: float = 1000.0,
    num_vehicles: int = 1000,
    crs: str = "EPSG:32643",
    seed: int = None
) -> list:
    """
    Creates `num_vehicles` realistic (start, destination) pairs for the
    mapped Jaipur district (or any OSM area, given centre + CRS).

    center : (lat, lon) the district is centred on; features are queried
             within `dist` metres of it (same footprint as the network).
    dist   : radius in metres for the feature query (default 1000).
    crs    : the graph's projected CRS (epsg string), used to inverse-project
             graph nodes for snapping (default EPSG:32643 - UTM 43N, Jaipur).

    Returns the standard vehicle contract list:
        [{"id": 1.., "start": node, "destination": node}, ...]
    """
    rng = random.Random(seed)
    lonlat = _lonlat_index(G, crs)

    print("OSM land-use demand trace:")
    origin_nodes = _category_nodes(G, center, dist, crs, lonlat, _ORIGIN_TAGS,
                                 "residential origins")
    dest_nodes = _category_nodes(G, center, dist, crs, lonlat, _DESTINATION_TAGS,
                               "education/office/commercial destinations")

    vehicles = []
    pool = set(origin_nodes) | set(dest_nodes)
    while len(vehicles) < num_vehicles:
        start = rng.choice(origin_nodes)
        candidates = [n for n in dest_nodes if n != start] or [n for n in pool if n != start]
        if not candidates:
            raise ValueError("Demand trace needs >= 2 distinct road nodes")
        destination = rng.choice(candidates)
        vehicles.append({
            "id": len(vehicles) + 1,
            "start": start,
            "destination": destination,
        })
    return vehicles